
export CUDA_VISIBLE_DEVICES=2
python -u evaluate.py \
--dataset AQuA \
--model GPT-j-6B \
--base_model EleutherAI/gpt-j-6b \
--weights_path ./checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_01/ \
--adapter LoRA \
--embedding_lambda 0.1 > scripts/evaluations_gpt-j-6b_lora_att_AQuA_16_3e-4_3_01.txt

python -u evaluate.py \
--dataset AQuA \
--model GPT-j-6B \
--base_model EleutherAI/gpt-j-6b \
--weights_path ./checkpoints/gpt-j-6b_bottleneck_att/math_10k/16_3e-4_3_01/ \
--adapter Bottleneck \
--embedding_lambda 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_AQuA_16_3e-4_3_01.txt

