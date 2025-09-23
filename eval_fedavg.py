import os
import numpy as np
from tqdm import tqdm

from utils.utils import seed_everything, get_data_dir, get_learner, get_aggregator
from utils.constants import EMBEDDING_DIM
from utils.args import FedAvgArgumentsManager
from utils.client_init import init_clients


def run(arguments_manager_):
    """
    Main Federated Averaging (baseline) pipeline
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

    print("==> Initializing clients...")
    clients = init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train")
        )     

    print("==> Initializing test clients...")
    test_clients = init_clients(
            args_,
            data_dir=os.path.join(data_dir, "test"),
            chkpts_dir=os.path.join(chkpts_dir, "test")
        )
    
    global_learner = get_learner(
        name=args_.experiment,
        device=args_.device,
        optimizer_name=args_.optimizer,
        scheduler_name=args_.lr_scheduler,
        initial_lr=args_.fedavg_lr,
        n_rounds=args_.n_rounds,
        input_dimension=EMBEDDING_DIM[args_.backbone]
    )

    aggregator = get_aggregator(
        aggregator_type=args_.aggregator_type, 
        clients=clients,
        algorithm=args_.algorithm,
        global_learner=global_learner,
        sampling_rate=args_.sampling_rate,
        log_freq=args_.log_freq,
        test_clients=test_clients,
        verbose=args_.verbose,
        seed=args_.seed
    )
    
    all_client_results_ = []
    all_test_client_results_ = []
    all_eval_rounds_ = []

    aggregator.aggregate_metrics()
    print("Training with FedAvg...")
    for ii in tqdm(range(args_.n_rounds)):
        aggregator.mix()
        if (ii % args_.log_freq) == (args_.log_freq - 1):
            aggregator.save_state(chkpts_dir)
            aggregator.aggregate_metrics(epoch=ii, max_epochs=args_.n_rounds, save_path=results_dir)
    aggregator.save_state(chkpts_dir)

    all_client_results_ = np.array(all_client_results_)
    all_test_client_results_ = np.array(all_test_client_results_)
    all_eval_rounds_ = np.array(all_eval_rounds_)
    
    np.save(os.path.join(results_dir, "fedavg_all_eval_rounds.npy"), all_eval_rounds_)
    np.save(os.path.join(results_dir, "fedavg_all_client_results.npy"), all_client_results_)
    np.save(os.path.join(results_dir, "fedavg_all_test_client_results.npy"), all_test_client_results_)


if __name__ == "__main__":
    arguments_manager = FedAvgArgumentsManager()
    arguments_manager.parse_arguments()
    run(arguments_manager)
    
