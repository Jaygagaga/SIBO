export CUDA_VISIBLE_DEVICES=0

python -u finetune.py \
--base_model EleutherAI/gpt-j-6b \
--data_path ./ft-training_set/math_10k.json \
--batch_size 16 \
--micro_batch_size 4 \
--num_epochs 3 \
--learning_rate 3e-4 \
--cutoff_len 256 \
--val_set_size 120 \
--eval_step 80 \
--save_step 80 \
--adapter_name lora \
--embedding_lambda 0.1 \
--ffn True \
--lora_r 32 \
--lora_alpha 64 \
--target_modules ["q_proj","k_proj","v_proj","fc_in","fc_out"] \
--output_dir ./checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_01/ > gpt-j-6b_lora_att_math_10k_16_3e-4_3_01.txt

python -u finetune.py \
--base_model EleutherAI/gpt-j-6b \
--data_path ./ft-training_set/math_10k.json \
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
--embedding_lambda 0.3 \
--ffn True \
--target_modules ["fc_out"] \
--output_dir ./checkpoints/gpt-j-6b_bottleneck_att/math_10k/16_3e-4_3_03/ > gpt-j-6b_bottleneck_att_math_10k_16_3e-4_3_03.txt
#python -u finetune.py \
#--base_model EleutherAI/gpt-j-6b \
#--data_path ./ft-training_set/math_10k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name lora \
#--embedding_lambda 0.2 \
#--att False \
#--ffn True \
#--lora_r 32 \
#--lora_alpha 64 \
#--target_modules ["q_proj","k_proj","v_proj","fc_in","fc_out"] \
#--output_dir ./checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_02/ > gpt-j-6b_lora_att_math_10k_16_3e-4_3_02.txt
#python -u finetune.py \
#--base_model EleutherAI/gpt-j-6b \
#--data_path ./ft-training_set/math_10k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name lora \
#--embedding_lambda 0.3 \
#--att False \
#--ffn True \
#--lora_r 32 \
#--lora_alpha 64 \
#--target_modules ["q_proj","k_proj","v_proj","fc_in","fc_out"] \
#--output_dir ./checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_03/ > gpt-j-6b_lora_att_math_10k_16_3e-4_3_03.txt


