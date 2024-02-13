export CUDA_VISIBLE_DEVICES=2
#export TRANSFORMERS_VERBOSITY=warning
#export BASE_MODEL=/root/autodl-tmp/gpt-j-6b
#export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/LLM-Adapters1/transformers/src"

python -u finetune.py \
--base_model EleutherAI/gpt-j-6b \
--data_path ./ft-training_set/commonsense_170k.json \
--batch_size 16 \
--micro_batch_size 4 \
--num_epochs 3 \
--learning_rate 3e-4 \
--cutoff_len 256 \
--val_set_size 120 \
--eval_step 80 \
--save_step 80 \
--adapter_name lora \
--embedding_lambda 0.3 \
--ffn True \
--lora_r 32 \
--lora_alpha 64 \
--target_modules ["q_proj","k_proj","v_proj","fc_in","fc_out"] \
--output_dir ./checkpoints/gpt-j-6b_lora_att/commonsense_170k/16_3e-4_3_03/ > gpt-j-6b_lora_att_commonsense_170k_16_3e-4_3_03.txt

python -u finetune.py \
--base_model EleutherAI/gpt-j-6b \
--data_path ./ft-training_set/commonsense_170k.json \
--batch_size 16 \
--micro_batch_size 4 \
--num_epochs 3 \
--learning_rate 3e-4 \
--cutoff_len 256 \
--val_set_size 120 \
--eval_step 80 \
--save_step 80 \
--adapter_name bottleneck \
--bottleneck_size 256 \
--embedding_lambda 0.1 \
--ffn True \
--bottleneck_size 256 \
--target_modules ["fc_out"] \
--output_dir ./checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ > gpt-j-6b_bottleneck_att_commonsense_170k_16_3e-4_3_01.txt
