SEEDS=(3407 12345 202502)
client_generation_seed=202502

s_frac=1.0
alphas=(0.3) 
split_method="by_labels_split" 
n_clients_values=(10)

backbone="base_patch14_dinov2" 
val_frac=0.2 
device="cuda"
classifier="linear"
classifier_optimizer="adam"  
learning_rates=(0.001)
local_epochs=100

anonymizers=("cvae_fedavg") # "cgan_fedavg"
n_fedavg_rounds_values=(50)
cvae_beta=0.1
cvae_var=1.0
generated_factor=1.0

# Differential Privacy
max_grad_norm=1.5 
epsilons=(1.0) 
delta=1e-4 

for n_clients in "${n_clients_values[@]}"; do
    for alpha in "${alphas[@]}"; do
        for anonymizer in "${anonymizers[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for n_fedavg_rounds in "${n_fedavg_rounds_values[@]}"; do
                    for epsilon in "${epsilons[@]}"; do
                        # Get the current date and time
                        current_time=$(date +"%Y_%m_%d_%H_%M")

                        base_chkpts_dir="chkpts/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/${anonymizer}/${current_time}" 
                        base_results_dir="results/organsmnist/$backbone/s_frac=${s_frac}/${n_clients}_clients/${split_method}/alpha=${alpha}/${anonymizer}/${current_time}"

                        results_all_seeds_file="${base_results_dir}/results_all_seeds.txt"
                        echo "Experiment results log - Generated on ${current_time}" 
                        
                        for SEED in "${SEEDS[@]}"; do
                            echo "=> Generate data with seed $client_generation_seed"
                            
                            rm -rf data/organsmnist/all_clients_data
                            
                            python data/organsmnist/generate_data.py \
                                --n_clients $n_clients \
                                --split_method $split_method \
                                --n_components -1 \
                                --alpha $alpha \
                                --s_frac $s_frac \
                                --test_clients_frac 0.0 \
                                --test_data_frac 0.2 \
                                --seed $client_generation_seed

                            echo "=> Evaluate Ours method using an anonymizer model (DP-CVAE/-CGAN) trained with FedAvg"

                            current_seed_time=$(date +"%Y_%m_%d_%H_%M")
                            chkpts_dir="${base_chkpts_dir}/${current_seed_time}"  
                            results_dir="${base_results_dir}/seed=${SEED}-chkpt=${current_seed_time}"
                            
                            python eval_ours.py \
                                organsmnist \
                                --backbone $backbone \
                                --client_type personalized \
                                --val_frac $val_frac \
                                --bz 128 \
                                --device $device \
                                --seed $SEED \
                                --chkpts_dir $chkpts_dir \
                                --results_dir $results_dir \
                                --verbose 1 \
                                --n_fedavg_rounds $n_fedavg_rounds \
                                --classifier $classifier \
                                --classifier_optimizer $classifier_optimizer \
                                --local_epochs $local_epochs \
                                --weights_grid_resolution 0.1 \
                                --anonymizer $anonymizer \
                                --lr $lr \
                                --cvae_beta $cvae_beta \
                                --cvae_var $cvae_var \
                                --generated_factor $generated_factor \
                                --enable_dp \
                                --max_grad_norm $max_grad_norm \
                                --epsilon $epsilon \
                                --delta $delta
                            
                            echo "----------------------------------------------------"

                            # Extract best metrics and best weight, save to results
                            BEST_ACC=$(grep "Best accuracy in grid search evaluation:" "${results_dir}/results.log" | awk '{print $NF}')
                            BEST_BALANCED_ACC=$(grep "Best balanced accuracy in grid search evaluation:" "${results_dir}/results.log" | awk '{print $NF}')
                            BEST_WEIGHT=$(grep "Optimal weight for accuracy:" "${results_dir}/results.log" | awk '{print $NF}')
                            BEST_WEIGHT_BALANCED=$(grep "Optimal weight for balanced accuracy:" "${results_dir}/results.log" | awk '{print $NF}')
                            AVG_TEST_ACC=$(grep "Test accuracy using optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')
                            AVG_TEST_BALANCED_ACC=$(grep "Test balanced accuracy using optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')
                            AVG_TEST_F1_MACRO=$(grep "Test F1 score (Macro) using optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}') 
                            AVG_TEST_F1_WEIGHTED=$(grep "Test F1 score (Weighted) using optimal weight" "${results_dir}/results.log" | awk -F': ' '{print $2}' | awk '{print $1}')

                            echo "Best accuracy in grid search evaluation for seed $SEED: $BEST_ACC %"
                            echo "Best balanced accuracy in grid search evaluation for seed $SEED: $BEST_BALANCED_ACC %"
                            echo "Optimal weight for accuracy for seed $SEED: $BEST_WEIGHT"
                            echo "Optimal weight for balanced accuracy for seed $SEED: $BEST_WEIGHT_BALANCED"
                            echo "Average test accuracy for seed $SEED: $AVG_TEST_ACC %"
                            echo "Average test balanced accuracy for seed $SEED: $AVG_TEST_BALANCED_ACC %"
                            echo "Average test F1 score (Macro) for seed $SEED: $AVG_TEST_F1_MACRO" 
                            echo "Average test F1 score (Weighted) for seed $SEED: $AVG_TEST_F1_WEIGHTED" 

                            echo "$BEST_ACC $BEST_WEIGHT $BEST_BALANCED_ACC $BEST_WEIGHT_BALANCED $AVG_TEST_ACC $AVG_TEST_BALANCED_ACC $AVG_TEST_F1_MACRO $AVG_TEST_F1_WEIGHTED" >> "$results_all_seeds_file" 

                            echo "----------------------------------------------------"
                        done
                    
                        echo "=> Aggregate results across all seeds"
                        
                        python utils/utils_ours.py \
                            --results_dir "${base_results_dir}" \
                            --seeds ${SEEDS[@]} >> "$results_all_seeds_file"

                        echo "----------------------------------------------------"
                    done
                done
            done
        done
    done
done