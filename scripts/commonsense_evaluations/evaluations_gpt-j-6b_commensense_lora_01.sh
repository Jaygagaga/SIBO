export CUDA_VISIBLE_DEVICES=0


python -u commonsense_evaluate.py \
--dataset piqa \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_piqa_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset boolq \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_boolq_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset social_i_qa \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_social_i_qa_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset hellaswag \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_hellaswag_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset winogrande \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_winogrande_4_3e-4_3_01.txt


python -u commonsense_evaluate.py \
--dataset ARC-Challenge \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_ARC-Challenge_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset ARC-Easy \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_ARC-Easy_4_3e-4_3_01.txt

python -u commonsense_evaluate.py \
--dataset openbookqa \
--model GPT-j-6B \
--base_model /root/autodl-tmp/gpt-j-6b \
--lora_weights /root/autodl-tmp/LLM-Adapters2/checkpoints/gpt-j-6b_bottleneck_att/commonsense_170k/16_3e-4_3_01/ \
--adapter Bottleneck \
--batch_size 1 \
--embedding_alpha 0.1 > scripts/evaluations_gpt-j-6b_bottleneck_att_openbookqa_4_3e-4_3_01.txt
