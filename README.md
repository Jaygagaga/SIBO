<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


<h3 align="center">
    <p>SIBO: A Simple Booster for Parameter-Efficient Fine-Tuning </p>
</h3>

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)


## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set environment variables, or modify the files referencing `BASE_MODEL`:

```bash
# Files referencing `BASE_MODEL`
# export_hf_checkpoint.py
# export_state_dict_checkpoint.py

export BASE_MODEL=EleutherAI/gpt-j-6b
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

## Datasets
Datsets for training and evaluations can be downloaded from https://github.com/AGI-Edgerunners/LLM-Adapters. 
We used `math_10k.json` and `commonsense_170k.json` under ft-training_set folder in LLM-Adapters repository. Datasets can be replaced in training scripts.  

## Training(finetune.py)


The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). `EleutherAI/gpt-j-6b` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
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
  --output_dir ./checkpoints/gpt-j-6b_lora_att/math_10k/16_3e-4_3_01/
```

To use the AdapterH, just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
```

## Inference (generate.py)

This file reads the foundation model from the Hugging Face model hub and finetuned model weights from `checkpoints` directory, e.g.`'./checkpoints/gpt-j-6b_bottleneck_att/math_10k/16_3e-4_3_01'` , and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun generate.py \
    --base_model 'EleutherAI/gpt-j-6b' \
    --weights_path ./checkpoints/gpt-j-6b_bottleneck_att/math_10k/16_3e-4_3_01/
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model GPT-j-6B \  #specify the base model ['LLaMA-7B','LLaMA-13B', 'GPT-j-6B']
    --base_model 'EleutherAI/gpt-j-6b' \
    --dataset SVAMP \  #specify the test dataset
    --adapter LoRA \   #specify the adapter name ["LoRA", "Bottleneck"]
    --weights_path ./checkpoints/gpt-j-6b_bottleneck_att/math_10k/16_3e-4_3_01/ \  #specify the path to finetuned weights
    --embedding_lambda 0.1  #specify embedding lambda
```

<!-- ## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_10k.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256

Models: LLaMA-13B, LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-7B-LoRA         | 4.2M                 | 18GB          |     4h           | 
| LLaMA-7B-AdapterH     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-AdapterP     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     4h           |  -->


