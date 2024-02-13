export CUDA_VISIBLE_DEVICES=1
#export TRANSFORMERS_VERBOSITY=warning
export BASE_MODEL=/root/autodl-tmp/llama-7b-hf
#export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/LLM-Adapters1/transformers/src"

python -u finetune.py \
--base_model /root/autodl-tmp/llama-7b-hf \
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
--embedding_alpha 0.1 \
--att False \
--ffn True \
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_01/ > llama_7b_lora_att_commonsense_170k_16_3e-4_3_01.txt
#
python -u finetune.py \
--base_model /root/autodl-tmp/llama-7b-hf \
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
--embedding_alpha 0.2 \
--att False \
--ffn True \
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_02/ > llama_7b_lora_att_commonsense_170k_16_3e-4_3_02.txt

python -u finetune.py \
--base_model /root/autodl-tmp/llama-7b-hf \
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
--embedding_alpha 0.3 \
--att False \
--ffn True \
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_03/ > llama_7b_lora_att_commonsense_170k_16_3e-4_3_03.txt

python -u finetune.py \
--base_model /root/autodl-tmp/llama-7b-hf \
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
--embedding_alpha 0.4 \
--att False \
--ffn True \
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_04/ > llama_7b_lora_att_commonsense_170k_04_16_3e-4_3_04.txt

python -u finetune.py \
--base_model /root/autodl-tmp/llama-7b-hf \
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
--embedding_alpha 0.5 \
--att False \
--ffn True \
--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_05/ > llama_7b_lora_att_commonsense_170k_04_16_3e-4_3_05.txt

#python -u finetune.py \
#--base_model /root/autodl-tmp/llama-7b-hf \
#--data_path ./ft-training_set/commonsense_170k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name lora \
#--embedding_alpha 0.6 \
#--att False \
#--ffn True \
#--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
#--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_06/ > llama_7b_lora_att_commonsense_170k_04_16_3e-4_3_06.txt
#
#python -u finetune.py \
#--base_model /root/autodl-tmp/llama-7b-hf \
#--data_path ./ft-training_set/commonsense_170k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name lora \
#--embedding_alpha 0.7 \
#--att False \
#--ffn True \
#--target_modules ["q_proj","k_proj","v_proj","up_proj","down_proj"] \
--lora_r 32 \
--lora_alpha 64 \
#--output_dir ./checkpoints/llama_7b_lora_att/commonsense_170k/16_3e-4_3_07/ > llama_7b_lora_att_commonsense_170k_04_16_3e-4_3_07.txt
