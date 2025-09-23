SEEDS=(3407 12345 202502)
client_generation_seed=202502

s_frac=4600 # Fixed number of total samples used

backbone="base_patch14_dinov2" 
val_frac=0.2 
client_type="normal"
aggregator_type="centralized"
algorithms=("normal") 
fedavg_learning_rates=(0.001)
num_rounds=(50)
lr_scheduler="multi_step"
device="cuda"

for algorithm in "${algorithms[@]}"; do
    for fedavg_lr in "${fedavg_learning_rates[@]}"; do
        for n_rounds in "${num_rounds[@]}"; do
            # Get the current date and time in the format YYYY_MM_DD_HH_MM
            current_time=$(date +"%Y_%m_%d_%H_%M")

            base_chkpts_dir="chkpts/camelyon17/$backbone/s_frac=${s_frac}/fedavg/${current_time}" 
            base_results_dir="results/camelyon17/$backbone/s_frac=${s_frac}/fedavg/${current_time}"
                    
            results_all_seeds_file="${base_results_dir}/results_all_seeds.txt"
            echo "Experiment results log - Generated on ${current_time}"

            for SEED in "${SEEDS[@]}"; do
                echo "=> Generate data with seed $client_generation_seed"

                rm -rf data/camelyon17/all_clients_data

                python data/camelyon17/generate_data.py \
                    --s_frac $s_frac \
                    --n_clients 5 \
                    --test_clients_frac 0.0 \
                    --test_data_frac 0.2 \
                    --val_data_frac $val_frac \
                    --seed $client_generation_seed

                echo "=> Train global model with FedAvg"

                current_seed_time=$(date +"%Y_%m_%d_%H_%M") 
                chkpts_dir="${base_chkpts_dir}/${current_seed_time}"  
                results_dir="${base_results_dir}/seed=${SEED}-chkpt=${current_seed_time}"

                python eval_fedavg.py \
                    camelyon17 \
                    --backbone $backbone \
                    --client_type $client_type \
                    --val_frac $val_frac \
                    --bz 128 \
                    --device $device \
                    --seed $SEED  \
                    --chkpts_dir $chkpts_dir \
                    --results_dir $results_dir \
                    --verbose 1 \
                    --aggregator_type $aggregator_type \
                    --algorithm $algorithm \
                    --n_rounds $n_rounds \
                    --log_freq 10 \
                    --optimizer sgd \
                    --fedavg_lr $fedavg_lr \
                    --lr_scheduler $lr_scheduler

                echo "----------------------------------------------------"
            done

            echo "=> Aggregate results across all seeds"

            python3 utils/utils_baseline.py \
                --results_dir "${base_results_dir}" \
                --seeds ${SEEDS[@]} >> "$results_all_seeds_file"

            echo "----------------------------------------------------"
        done
    done
done
