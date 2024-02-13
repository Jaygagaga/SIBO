
export CUDA_VISIBLE_DEVICES=2
#python -u evaluate.py \
#--dataset AQuA \
#--model GPT-j-6B \
#--base_model /root/autodl-tmp/gpt-j-6b \
#--lora_weights /root/autodl-tmp/LLM-Adapters/checkpoints/gpt-j-6b1_lora_att/math_10k/16_3e-4_3_01/ \
#--adapter LoRA \
#--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_lora_att1_AQuA_16_3e-4_3_01.txt &
#
#python -u evaluate.py \
#--dataset AQuA \
#--model GPT-j-6B \
#--base_model /root/autodl-tmp/gpt-j-6b \
#--lora_weights /root/autodl-tmp/LLM-Adapters/checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_02/ \
#--adapter LoRA \
#--embedding_alpha 0.2 > scripts/evaluations_gpt-j-6b_lora_att1_AQuA_16_3e-4_3_02.txt

#python -u evaluate.py \
#--dataset AQuA \
#--model LLaMA-7B \
#--base_model /root/autodl-tmp/gpt-j-6b \
#--lora_weights /root/autodl-tmp/LLM-Adapters/checkpoints/gpt-j-6b1_lora_att/math_10k/16_3e-4_3_03/ \
#--adapter LoRA \
#--embedding_alpha 0.3 > scripts/evaluations_gpt-j-6b_lora_att1_AQuA_16_3e-4_3_03.txt

#python -u evaluate.py \
#--dataset mawps \
#--model GPT-j-6B \
#--base_model /root/autodl-tmp/gpt-j-6b \
#--lora_weights /root/autodl-tmp/LLM-Adapters/checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_02/ \
#--adapter LoRA \
#--embedding_alpha 0.2 > scripts/evaluations_gpt-j-6b_lora_att1_mawps_16_3e-4_3_02.txt
#
python -u evaluate.py \
--dataset AQuA \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters/checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_03/ \
--adapter LoRA \
--embedding_alpha 0.3 > scripts/evaluations_gpt-j-6b_lora_att1_AQuA_16_3e-4_3_03.txt

