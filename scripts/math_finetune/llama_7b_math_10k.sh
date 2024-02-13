export CUDA_VISIBLE_DEVICES=1

python -u finetune.py \
--base_model yahma/llama-7b-hf \
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
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--output_dir ./checkpoints/llama_7b_lora_att/math_10k/16_3e-4_3_01/ > llama_7b_lora_att_math_10k_16_3e-4_3_01.txt

python -u finetune.py \
--base_model yahma/llama-7b-hf \
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
--embedding_lambda 0.1 \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir ./checkpoints/llama_7b_bottleneck_att/math_10k/16_3e-4_3_01/ > llama_7b_bottleneck_math_10k_16_3e-4_3_01.txt

#--output_dir ./checkpoints/llama_7b_lora_att/math_10k/16_3e-4_3_03/ > llama_7b_lora_att_math_10k_16_3e-4_3_03.txt
