export CUDA_VISIBLE_DEVICES=0


python -u commonsense_evaluate.py \
--dataset hellaswag \
--model LLaMA-7B \
--base_model yahma/llama-7b-hf \
--weights_path ./checkpoints/llama-7b-hf_lora_att/commonsense_170k/16_3e-4_3_01/ \
--adapter LoRA \
--batch_size 1 \
--embedding_lambda 0.1 > scripts/llama-7b-hf_lora_att_hellaswag_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset hellaswag \
--model LLaMA-7B \
--base_model yahma/llama-7b-hf \
--weights_path ./checkpoints/llama-7b-hf_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_lambda 0.1 > scripts/llama-7b-hf_bottleneck_att_hellaswag_4_3e-4_3_01.txt
