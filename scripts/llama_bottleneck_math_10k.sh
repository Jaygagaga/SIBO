export CUDA_VISIBLE_DEVICES=0
#export TRANSFORMERS_VERBOSITY=warning
export BASE_MODEL=yahma/llama-7b-hf
export PYTHONPATH="${PYTHONPATH}:/home/wzh/AdapterHub/adapter-transformers/src"

#python -u finetune.py \
#--base_model yahma/llama-7b-hf \
#--data_path ./ft-training_set/math_10k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name bottleneck \
#--embedding_alpha 0.1 \
#--att False \
#--ffn True \
#--target_modules ["up_proj","gate_proj"] \
#--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_01/ > llama_bottleneck_math_10k_16_3e-4_3_01.txt
##
#python -u finetune.py \
#--base_model yahma/llama-7b-hf \
#--data_path ./ft-training_set/math_10k.json \
#--batch_size 16 \
#--micro_batch_size 4 \
#--num_epochs 3 \
#--learning_rate 3e-4 \
#--cutoff_len 256 \
#--val_set_size 120 \
#--eval_step 80 \
#--save_step 80 \
#--adapter_name bottleneck \
#--embedding_alpha 0.2 \
#--att False \
#--ffn True \
#--target_modules ["up_proj","gate_proj"] \
#--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_02/ > llama_bottleneck_math_10k_16_3e-4_3_02.txt

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
--embedding_alpha 0.3 \
--att False \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_03/ > llama_bottleneck_math_10k_16_3e-4_3_03.txt

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
--embedding_alpha 0.4 \
--att False \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir checkpoints/llama_bottleneck_att/16_3e-4_3_04/ > llama_bottleneck_math_10k_04_16_3e-4_3_04.txt

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
--embedding_alpha 0.5 \
--att False \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_05/ > llama_bottleneck_math_10k_04_16_3e-4_3_05.txt

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
--embedding_alpha 0.6 \
--att False \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_06/ > llama_bottleneck_math_10k_04_16_3e-4_3_06.txt

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
--embedding_alpha 0.7 \
--att False \
--ffn True \
--target_modules ["up_proj","gate_proj"] \
--output_dir ./checkpoints/llama_bottleneck_att/16_3e-4_3_07/ > llama_bottleneck_math_10k_04_16_3e-4_3_07.txt
