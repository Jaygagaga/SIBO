export CUDA_VISIBLE_DEVICES=2
python -u evaluate.py \
--dataset AQuA \
--model LLaMA-7B \
--base_model yahma/llama-7b-hf \
--lora_weights ./checkpoints/llama-7b-hf_lora_att/math_10k/16_3e-4_3_01/ \
--adapter LoRA \
--embedding_alpha 0.1 > scripts/llama-7b-hf_lora_att_AQuA_16_3e-4_3_01.txt &