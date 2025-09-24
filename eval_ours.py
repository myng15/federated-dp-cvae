import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from geomloss import SamplesLoss

from utils.utils import seed_everything, get_data_dir
from utils.constants import EMBEDDING_DIM
from utils.args import OursArgumentsManager
from utils.client_init import init_clients


# ---------- Helpers: Anonymizer training (CVAE-FEDAVG / CGAN-FEDAVG) ----------

def train_anonymizer_fedavg(clients, args_):
    """
    Train CVAE/CGAN via FedAvg
    """
    from anonymizer.cvae import CVAEAnonymizer
    from anonymizer.cgan import CGANAnonymizer
    from aggregator import CVAEAggregator, CGANAggregator

    log_freq = getattr(args_, "log_freq", 10)
    chkpts_dir = os.path.join(args_.chkpts_dir, args_.anonymizer)
    os.makedirs(chkpts_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(args_.seed)

    if args_.anonymizer == "cvae_fedavg":
        anonymizer = CVAEAnonymizer(args=args_, g=g)
    elif args_.anonymizer == "cgan_fedavg":
        anonymizer = CGANAnonymizer(args=args_, g=g)
    
    # Initialize global CVAE/CGAN model
    global_trainer = anonymizer.get_trainer(client=None, is_trained=False)
    # Initialize a local CVAE/CGAN for each client
    for client in clients:
        client.trainer = anonymizer.get_trainer(client=client, is_trained=False)
    
    if args_.anonymizer == "cvae_fedavg":
        aggregator = CVAEAggregator(
            clients=clients,
            global_trainer=global_trainer,
            anonymizer=anonymizer,
            log_freq=log_freq,
            verbose=args_.verbose,
            seed=args_.seed,
        )
    else:
        aggregator = CGANAggregator(
            clients=clients,
            global_trainer=global_trainer,
            anonymizer=anonymizer,
            log_freq=log_freq,
            verbose=args_.verbose,
            seed=args_.seed,
        )

    # Run FedAvg rounds
    for round_idx in tqdm(range(args_.n_fedavg_rounds)):
        aggregator.mix()
        if (round_idx % log_freq) == (log_freq - 1):
            aggregator.aggregate_metrics(anonymizer)

    # Save final state of global trainer (and optionally clients')
    aggregator.save_state(chkpts_dir)


# ---------- Helpers: Grid search and evaluation ----------

def eval_client_grid(client_, weights_grid_):
    """
    Run grid search on validation set (per client) to find the optimal local model weight (λ)

    :param client_: Client instance
    :param weights_grid_: list of weights (λ) to evaluate
    :return: acc_scores, balanced_acc_scores, f1_scores_macro, f1_scores_weighted
    """
    acc_scores = np.zeros((len(weights_grid_)))
    balanced_acc_scores = np.zeros((len(weights_grid_)))
    f1_scores_macro = np.zeros((len(weights_grid_)))
    f1_scores_weighted = np.zeros((len(weights_grid_)))
        
    for i, weight in enumerate(weights_grid_):
        client_acc, client_balanced_acc, client_f1_macro, client_f1_weighted = client_.evaluate(weight, val_mode=True) 
        acc_scores[i] = client_acc * client_.n_val_samples
        balanced_acc_scores[i] = client_balanced_acc * client_.n_val_samples
        f1_scores_macro[i] = client_f1_macro * client_.n_val_samples 
        f1_scores_weighted[i] = client_f1_weighted * client_.n_val_samples 

    return acc_scores, balanced_acc_scores, f1_scores_macro, f1_scores_weighted 

def eval_client_test(client_, weight_):
    """
    Evaluate client on test set for a given weight

    :param client_: Client instance
    :param weights_grid_: weight (λ) to evaluate
    :return: acc_score, balanced_acc_score, f1_score_macro, f1_score_weighted
    """
    # Evaluate the client with the selected weight (λ)
    client_acc, client_balanced_acc, client_f1_macro, client_f1_weighted = client_.evaluate(weight_, val_mode=False)
    acc_score = client_acc * client_.n_test_samples
    balanced_acc_score = client_balanced_acc * client_.n_test_samples
    f1_score_macro = client_f1_macro * client_.n_test_samples 
    f1_score_weighted = client_f1_weighted * client_.n_test_samples 

    return acc_score, balanced_acc_score, f1_score_macro, f1_score_weighted


# ---------- Helpers: Metrics ----------

def compute_wasserstein_distance(client, anonymized_features, anonymized_labels, embeddings_save_path):
    """
    Compute Wasserstein distance (via Sinkhorn divergence) between real and generated embeddings
    Save real/generated embeddings if a save path is provided and data is non-empty

    :param client: Client instance (expects client.train_features etc. as numpy arrays)
    :param anonymized_features: numpy array of anonymized (generated) features (N, D)
    :param anonymized_labels: numpy array of the anonymized features' corresponding labels (N,)
    :param embeddings_save_path: path to save the real and generated embeddings
    :return: (float) Wasserstein distance between real and generated embeddings
    """
    if len(anonymized_features) == 0:
        raise ValueError("No anonymized features generated for this client, cannot compute Wasserstein distance.")

    # Save real and generated embeddings for later evaluation
    client_embeddings = {
        'real_embeddings': client.train_features,
        'real_labels': client.train_labels,
        'real_val_embeddings': client.val_features,
        'real_val_labels': client.val_labels,
        'real_test_embeddings': client.test_features,
        'real_test_labels': client.test_labels,
        'generated_embeddings': np.array(anonymized_features),
        'generated_labels': np.array(anonymized_labels)
    }

    os.makedirs(embeddings_save_path, exist_ok=True)
    np.savez(os.path.join(embeddings_save_path,f'client_{client.id}.npz'), **client_embeddings)

    # Compute Wasserstein distance via Sinkhorn divergence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_embeddings = torch.tensor(client.train_features, dtype=torch.float32, device=device)
    generated_embeddings = torch.tensor(np.array(anonymized_features), dtype=torch.float32, device=device)

    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    wasserstein_distance = sinkhorn_loss(real_embeddings, generated_embeddings).item()
    wasserstein_distance = float(np.sqrt(wasserstein_distance))

    return wasserstein_distance

def compute_metric_statistics(all_scores, n_test_samples, is_accuracy):
    average = np.sum(all_scores) / np.sum(n_test_samples) * 100 if is_accuracy else np.sum(all_scores) / np.sum(n_test_samples)
    individual_scores = (all_scores / n_test_samples) * 100 if is_accuracy else (all_scores / n_test_samples)
    mean = np.mean(individual_scores)
    std = np.std(individual_scores)
    return average, mean, std


# ---------- Main pipeline ----------

def run(arguments_manager_):
    """
    Main Personalized Federated Learning pipeline, used for FedLambda, DP-CVAE (Ours) and DP-CGAN
    - Initialize clients
    - For DP-CVAE/DP-CGAN: Train anonymizer via FedAvg; Generate anonymized (differentially private) synthetic data using the trained anonymizer model
    - Interpolate local and global models to produce final predictions
    - Evaluate on validation set to pick best local model weight (λ)
    - Evaluate on test set using best weight (λ)
    """
    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()
    args_ = arguments_manager_.args
    
    seed_everything(args_.seed)

    data_dir = get_data_dir(args_.experiment)
    chkpts_dir = args_.chkpts_dir
    results_dir = args_.results_dir
    os.makedirs(chkpts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    weights_grid_ = np.arange(0.0, 1.0 + 1e-6, args_.weights_grid_resolution)

    print("===> Initializing clients...")
    features_dimension = EMBEDDING_DIM[args_.backbone] 
    clients = init_clients(
        args_,
        data_dir=os.path.join(data_dir, "train"),
        chkpts_dir=os.path.join(chkpts_dir, "train"),
        features_dimension=features_dimension,
        for_ours=True
    )
    
    print("===> Training anonymizer (DP-CVAE/-CGAN) with FedAvg...")
    if args_.anonymizer in ["cvae_fedavg", "cgan_fedavg"]:
        train_anonymizer_fedavg(clients, args_)
    
    print("===> Evaluating on validation set...")
    all_client_ids_ = []
    all_acc_scores_ = []
    all_balanced_acc_scores_ = []
    all_f1_scores_macro_ = []
    all_f1_scores_weighted_ = []
    n_train_samples_ = []
    n_val_samples_ = []
    n_integrated_global_samples_ = []
    all_wasserstein_dist_ = []
    
    for client in tqdm(clients):
        all_client_ids_.append(client.id)
        if args_.anonymizer in ["cvae_fedavg", "cgan_fedavg"]:
            anonymized_features, anonymized_labels = client.anonymize_data() 
            client.integrate_global_data(anonymized_features, anonymized_labels) 
            n_integrated_global_samples_.append(
                getattr(client, "global_features", np.array([])).shape[0] if hasattr(client, "global_features") else 0
            )
            # Compute Wasserstein distance between real and generated embeddings
            embeddings_save_path = os.path.join(results_dir, 'saved_embeddings')
            all_wasserstein_dist_.append(compute_wasserstein_distance(client, anonymized_features, anonymized_labels, embeddings_save_path))
        else:
            anonymized_features, anonymized_labels = [], []

        # Tune local model weight (λ) via grid search on validation set
        acc_scores, balanced_acc_scores, f1_scores_macro, f1_scores_weighted = eval_client_grid(client, weights_grid_) 
        all_acc_scores_.append(acc_scores)
        all_balanced_acc_scores_.append(balanced_acc_scores)
        all_f1_scores_macro_.append(f1_scores_macro) 
        all_f1_scores_weighted_.append(f1_scores_weighted) 
        n_train_samples_.append(client.n_train_samples)
        n_val_samples_.append(client.n_val_samples)

    all_client_ids_ = np.array(all_client_ids_)
    all_acc_scores_ = np.array(all_acc_scores_)
    all_balanced_acc_scores_ = np.array(all_balanced_acc_scores_)
    all_f1_scores_macro_ = np.array(all_f1_scores_macro_)
    all_f1_scores_weighted_ = np.array(all_f1_scores_weighted_)
    n_train_samples_ = np.array(n_train_samples_)
    n_val_samples_ = np.array(n_val_samples_)
    n_integrated_global_samples_ = np.array(n_integrated_global_samples_)
    all_wasserstein_dist_ = np.array(all_wasserstein_dist_)
    
    if len(all_wasserstein_dist_) > 0:
        avg_wasserstein_dist = np.mean(all_wasserstein_dist_)
        print(f"Overall average Wasserstein distance across clients: {avg_wasserstein_dist:.6f}")
    else:
        avg_wasserstein_dist = float('nan')
    
    # Aggregate validation metrics (average across clients) and select best weight (λ)
    normalized_acc_scores = np.nan_to_num(all_acc_scores_) / n_val_samples_[:, np.newaxis]
    acc_grid = normalized_acc_scores.mean(axis=0) * 100
    best_acc = np.max(acc_grid)
    best_idx = int(np.argmax(acc_grid))
    best_weight = weights_grid_[best_idx]
    print(f"Best accuracy: {best_acc:.2f} -> Optimal weight: {best_weight:.2f}")
    
    normalized_balanced_acc_scores = np.nan_to_num(all_balanced_acc_scores_) / n_val_samples_[:, np.newaxis]
    balanced_acc_grid = normalized_balanced_acc_scores.mean(axis=0) * 100
    best_balanced_acc = np.max(balanced_acc_grid)
    best_idx_balanced = int(np.argmax(balanced_acc_grid))
    best_weight_balanced = weights_grid_[best_idx_balanced]
    print(f"Best balanced accuracy: {best_balanced_acc:.2f} -> Optimal weight: {best_weight_balanced:.2f}")

    # Save the results to a seed-specific log file
    with open(os.path.join(results_dir, "results.log"), "a", buffering=1) as log_file:
        print(f"Best accuracy in grid search evaluation: {best_acc:.2f}", file=log_file)
        print(f"Best balanced accuracy in grid search evaluation: {best_balanced_acc:.2f}", file=log_file)
        print(f"Optimal weight for accuracy: {best_weight:.2f}", file=log_file)
        print(f"Optimal weight for balanced accuracy: {best_weight_balanced:.2f}", file=log_file)
        
    print("===> Evaluating on test set...")
    all_test_acc_scores_ = []
    all_test_balanced_acc_scores_ = []
    all_test_f1_scores_macro_ = [] 
    all_test_f1_scores_weighted_ = [] 
    n_test_samples_ = []

    for client in tqdm(clients):
        test_acc_score, test_balanced_acc_score, test_f1_macro, test_f1_weighted = eval_client_test(client, best_weight)
        all_test_acc_scores_.append(test_acc_score)
        all_test_balanced_acc_scores_.append(test_balanced_acc_score)
        all_test_f1_scores_macro_.append(test_f1_macro)
        all_test_f1_scores_weighted_.append(test_f1_weighted) 
        n_test_samples_.append(client.n_test_samples)
    
    all_test_acc_scores_ = np.array(all_test_acc_scores_) 
    all_test_balanced_acc_scores_ = np.array(all_test_balanced_acc_scores_) 
    all_test_f1_scores_macro_ = np.array(all_test_f1_scores_macro_) 
    all_test_f1_scores_weighted_ = np.array(all_test_f1_scores_weighted_) 
    n_test_samples_ = np.array(n_test_samples_)

    average_test_acc, mean_test_acc, std_test_acc = compute_metric_statistics(all_test_acc_scores_, n_test_samples_, is_accuracy=True)
    average_test_balanced_acc, mean_test_balanced_acc, std_test_balanced_acc = compute_metric_statistics(all_test_balanced_acc_scores_, n_test_samples_, is_accuracy=True)
    average_test_f1_macro, mean_test_f1_macro, std_test_f1_macro = compute_metric_statistics(all_test_f1_scores_macro_, n_test_samples_, is_accuracy=False)
    average_test_f1_weighted, mean_test_f1_weighted, std_test_f1_weighted = compute_metric_statistics(all_test_f1_scores_weighted_, n_test_samples_, is_accuracy=False)

    print(f'Average test accuracy: {average_test_acc:.2f}%')
    print(f'Mean and standard deviation of test accuracies across clients: Mean: {mean_test_acc:.2f}%, Std: {std_test_acc:.2f}%')
    print(f'Average test balanced accuracy: {average_test_balanced_acc:.2f}%')
    print(f'Mean and standard deviation of test balanced accuracies across clients: Mean: {mean_test_balanced_acc:.2f}%, Std: {std_test_balanced_acc:.2f}%')
    print(f'Average test F1 score (Macro): {average_test_f1_macro:.3f}')
    print(f'Mean and standard deviation of test F1 score (Macro) across clients: Mean: {mean_test_f1_macro:.3f}, Std: {std_test_f1_macro:.3f}')
    print(f'Average test F1 score (Weighted): {average_test_f1_weighted:.3f}')
    print(f'Mean and standard deviation of test F1 score (Weighted) across clients: Mean: {mean_test_f1_weighted:.3f}, Std: {std_test_f1_weighted:.3f}')

    with open(os.path.join(results_dir, "results.log"), "a", buffering=1) as log_file: 
        print(f"Test accuracy using optimal weight (Average | Mean | Std): {average_test_acc:.2f} | {mean_test_acc:.2f} | {std_test_acc:.2f}", file=log_file)
        print(f"Test balanced accuracy using optimal weight (Average | Mean | Std): {average_test_balanced_acc:.2f} | {mean_test_balanced_acc:.2f} | {std_test_balanced_acc:.2f}", file=log_file)
        print(f"Test F1 score (Macro) using optimal weight (Average | Mean | Std): {average_test_f1_macro:.3f} | {mean_test_f1_macro:.3f} | {std_test_f1_macro:.3f}", file=log_file)
        print(f"Test F1 score (Weighted) using optimal weight (Average | Mean | Std): {average_test_f1_weighted:.3f} | {mean_test_f1_weighted:.3f} | {std_test_f1_weighted:.3f}", file=log_file)

    results_dict = {
        "all_client_ids": all_client_ids_,
        "weights_grid": weights_grid_,
        "n_train_samples": n_train_samples_,
        "n_val_samples": n_val_samples_,
        "n_test_samples": n_test_samples_,
        "n_integrated_global_samples": n_integrated_global_samples_,
        "all_wasserstein_dist": all_wasserstein_dist_,
        "all_acc_scores": all_acc_scores_,
        "all_balanced_acc_scores": all_balanced_acc_scores_,
        "all_f1_scores_macro": all_f1_scores_macro_,
        "all_f1_scores_weighted": all_f1_scores_weighted_,
        "best_acc": best_acc,
        "best_weight": best_weight,
        "best_balanced_acc": best_balanced_acc,
        "best_weight_balanced": best_weight_balanced,
        "all_test_acc_scores": all_test_acc_scores_,
        "all_test_balanced_acc_scores": all_test_balanced_acc_scores_,
        "all_test_f1_scores_macro": all_test_f1_scores_macro_,
        "all_test_f1_scores_weighted": all_test_f1_scores_weighted_,
        "average_test_acc": average_test_acc,
        "mean_test_acc": mean_test_acc,
        "std_test_acc": std_test_acc,
        "average_test_balanced_acc": average_test_balanced_acc,
        "mean_test_balanced_acc": mean_test_balanced_acc,
        "std_test_balanced_acc": std_test_balanced_acc,
        "average_test_f1_macro": average_test_f1_macro,
        "mean_test_f1_macro": mean_test_f1_macro,
        "std_test_f1_macro": std_test_f1_macro,
        "average_test_f1_weighted": average_test_f1_weighted,
        "mean_test_f1_weighted": mean_test_f1_weighted,
        "std_test_f1_weighted": std_test_f1_weighted,
    }

    with open(os.path.join(results_dir, "results_dict.pkl"), 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    arguments_manager = OursArgumentsManager()
    arguments_manager.parse_arguments()
    run(arguments_manager)
    