import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import logging
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
sys.path.append(os.path.join(os.getcwd(), "transformers"))
import os
import re
import json
import torch
# from vllm import LLM, SamplingParams
import sys
from peft import PeftModel
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default='/root/autodl-tmp/llama-7b-hf')
parser.add_argument("--dataset_name", default='gsm8k')
parser.add_argument("--model_weights", default='')
parser.add_argument("--embedding_lambda", default=None)
parser.add_argument("--adapter", default=None)
parser.add_argument("--load_8bit", default=False)


args = parser.parse_args()


# dataset_name = "gsm8k"
# model_name = "vistral-7b"
# n_shot = 0


# sampling_params = SamplingParams(temperature=0.7, max_tokens=512, n=1, ) #top_p=0.95) temperature=0.7,
# sampling_params = SamplingParams(temperature=0.1, max_tokens=512,top_p=0.75,top_k=40, ) #top_p=0.95) temperature=0.7,

def reader(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

#
# def get_answer(sentence, dataset_name):
#     sentence = sentence.replace(',', '')
#     if dataset_name == 'AQuA':
#         pred = re.findall(r'The answer is \(([A-Z])\)', sentence)
#         if not pred:
#             return float('inf')
#         pred_answer = pred[-1]
#     else: #
#         pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
#         if not pred:
#             return float('inf')
#         pred_answer = float(pred[-1])
#         if isinstance(pred_answer, str):
#             try:
#                 pred_answer = float(pred_answer)
#             except ValueError as e:
#                 pred_answer = float('inf')
#     return pred_answer


def get_prompt_2shot(text):
    prompt = f"""
Reggie, Lynn, and Paisley ran together. Paisley ran 4 miles. Reggie ran 5 times what Paisley ran and 3 miles farther than Lynn. How many miles did Lynn run?
Please give the steps and the arabic numerals as the answer.
1. Paisley ran 4 miles.
2. Reggie ran 5 times what Paisley ran, which is 5 x 4 = 20 miles.
3. Reggie ran 20 miles.4. Lynn ran 3 miles less than Reggie, which is 20 - 3 = 17 miles.

Therefore, the answer is 17 miles.

{text}
Please give the steps and the arabic numerals as the answer.
"""
    return prompt


def get_prompt_zero_shot(text):
    prompt = f"""
{text}
Please give the steps and the arabic numerals as the answer.
"""
    return prompt


def get_vistral_prompt_zero_shot(text, tokenizer):
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    return prompt


def get_metamath_prompt_zero_shot(text):
    prompt = f"""
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{text}

### Response: Let's think step by step.
"""
    return prompt


def get_seallm_prompt(question):
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    include_end_instruct = True

    SYSTEM_PROMPT_1 = """You are a multilingual, helpful, respectful and honest assistant. You are built by DAMO Academy, Alibaba Group. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.
As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture.
"""

    """
    ```
        <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
    ```
    """
    text = ''
    end_instr = f" {E_INST}" if include_end_instruct else ""
    text += f"{BOS_TOKEN}{B_INST} {B_SYS} {SYSTEM_PROMPT_1} {E_SYS} Reggie, Lynn, and Paisley ran together. Paisley ran 4 miles. Reggie ran 5 times what Paisley ran and 3 miles farther than Lynn. How many miles did Lynn run?{end_instr}"
    text += f"1. Paisley ran 4 miles.\n2. Reggie ran 5 times what Paisley ran, which is 5 x 4 = 20 miles.\n3. Reggie ran 20 miles.4. Lynn ran 3 miles less than Reggie, which is 20 - 3 = 17 miles.\n\nTherefore, the answer is 17 miles. {EOS_TOKEN} "
    text += f"{BOS_TOKEN}{B_INST} If Jill can run up a hill at a speed of 9 feet/second and down the hill at a speed of 12 feet/second, what is the total time it takes her to run up and down a 900 foot hill?{end_instr}"
    text += f"To calculate the time it takes Jill to run up the hill, we divide the distance by her speed: 900 feet / 9 feet/second = 100 seconds.\nTo calculate the time it takes Jill to run down the hill, we divide the distance by her speed: 900 feet / 12 feet/second = 75 seconds.\nTo calculate the total time it takes Jill to run up and down the hill, we add the time it takes to run up and the time it takes to run down: 100 seconds + 75 seconds = 175 seconds.\n#### 175\nThe answer is: 175 {EOS_TOKEN}"

    text += f"{BOS_TOKEN}{B_INST} {question}{end_instr}"
    return text


def get_seallm_prompt_zero_shot(question):
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    include_end_instruct = True

    SYSTEM_PROMPT_1 = """You are a multilingual, helpful, respectful and honest assistant. You are built by DAMO Academy, Alibaba Group. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.
As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture.
"""

    """
    ```
        <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
    ```
    """
    text = ''
    end_instr = f" {E_INST}" if include_end_instruct else ""
    text += f"{BOS_TOKEN}{B_INST} {B_SYS} {SYSTEM_PROMPT_1} {E_SYS} {question}{end_instr}"

    return text


def get_seallm_prompt_internal(question):
    TURN_TEMPLATE = "<|im_start|>{role}\n{content}</s>"
    TURN_PREFIX = "<|im_start|>{role}\n"

    SYSTEM_PROMPT_1 = """You are a multilingual, helpful, respectful and honest assistant. You are built by DAMO Academy, Alibaba Group. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.
As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture."""
    text = ""
    text += TURN_TEMPLATE.format(role="system", content=SYSTEM_PROMPT_1)
    # demo 1
    text += TURN_TEMPLATE.format(role="user",
                                 content="Reggie, Lynn, and Paisley ran together. Paisley ran 4 miles. Reggie ran 5 times what Paisley ran and 3 miles farther than Lynn. How many miles did Lynn run?")
    text += TURN_TEMPLATE.format(role="assistant",
                                 content="1. Paisley ran 4 miles.\n2. Reggie ran 5 times what Paisley ran, which is 5 x 4 = 20 miles.\n3. Reggie ran 20 miles.4. Lynn ran 3 miles less than Reggie, which is 20 - 3 = 17 miles.\n\nTherefore, the answer is 17 miles.")
    # demo 2
    text += TURN_TEMPLATE.format(role="user",
                                 content="If Jill can run up a hill at a speed of 9 feet/second and down the hill at a speed of 12 feet/second, what is the total time it takes her to run up and down a 900 foot hill?")
    text += TURN_TEMPLATE.format(role="assistant",
                                 content="To calculate the time it takes Jill to run up the hill, we divide the distance by her speed: 900 feet / 9 feet/second = 100 seconds.\nTo calculate the time it takes Jill to run down the hill, we divide the distance by her speed: 900 feet / 12 feet/second = 75 seconds.\nTo calculate the total time it takes Jill to run up and down the hill, we add the time it takes to run up and the time it takes to run down: 100 seconds + 75 seconds = 175 seconds.\n#### 175\nThe answer is: 175")
    # test case
    text += TURN_TEMPLATE.format(role="user", content=question)
    text += TURN_PREFIX.format(role='assistant')

    return text


def get_seallm_prompt_internal_zero_shot(question):
    TURN_TEMPLATE = "<|im_start|>{role}\n{content}</s>"
    TURN_PREFIX = "<|im_start|>{role}\n"

    SYSTEM_PROMPT_1 = """You are a multilingual, helpful, respectful and honest assistant. You are built by DAMO Academy, Alibaba Group. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.
As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture."""
    text = ""
    text += TURN_TEMPLATE.format(role="system", content=SYSTEM_PROMPT_1)
    # test case
    text += TURN_TEMPLATE.format(role="user", content=question)
    text += TURN_PREFIX.format(role='assistant')

    return text


from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


def writer(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def process_outputs(outputs, test_data, dataset_name):
    try:
        test_data=test_data.remove_columns(['input_ids','attention_mask','labels'])
    except:
        test_data.pop('input_ids')
        test_data.pop('attention_mask')
        test_data.pop('labels')
    assert len(outputs) == len(test_data['instruction']), "outputs and training data should have the same length"
    # correct = 0
    test_data["generations"] = []
    for i in range(len(outputs)):
        # generation = {
        #     "generation": outputs[i],
        #     # "answer_pred": get_answer(outputs[i],dataset_name),
        # }
        test_data["generations"].append(outputs[i])

    return test_data


###test data
# test_data = reader(path)
# inputs = prepare_inputs(test_data)
def main():
    args = parser.parse_args()
    if "llama" in args.base_model:
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    cutoff_len = 256

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in args.base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in args.base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        return tokenized_full_prompt

    data_path = f"./dataset/{args.dataset_name}/test.json"
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    test_data = (
        data.shuffle().map(generate_and_tokenize_prompt)
    )
    # assert len(inputs) == len(test_data), "prompts and training data should have the same length"
    # print(f"---------------\nPrompt\n", inputs[0])
    # tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    device_map = "auto"
    if device == "cuda":
        if args.load_8bit:

            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                load_in_8bit=args.load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
                local_files_only=True,
            )
        model = PeftModel.from_pretrained(
            model,
            args.model_weights,
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            # instruction,
            # input=None,
            input_ids,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            adapter='lora',
            embedding_lambda=0.1,
            max_new_tokens=128,
            **kwargs,
    ):
        # prompt = generate_prompt(instruction, input)
        # inputs = tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            adapter=adapter,
            embedding_lambda=embedding_lambda,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # do_sample = True,
                # max_tokens=512
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    outputs = []
    test_data = test_data['train']
    test_data= test_data[:]
    for num, test in enumerate(test_data['input_ids']):
        # for test in test_data['train']['input_ids']:
        logger.info(f"No. {num} sample")
        input_ids = torch.tensor(test).unsqueeze(0).to(device)
        output = evaluate(input_ids, adapter=args.adapter,embedding_lambda = float(args.embedding_lambda))
        outputs.append(output)

    # # llm = LLM(model="meta-math/MetaMath-Mistral-7B", trust_remote_code=True, tensor_parallel_size=1)
    # llm = LLM(model=args.base_model, trust_remote_code=True, tensor_parallel_size=1)
    # model = PeftModel.from_pretrained(
    #     llm,
    #     args.model_weights,
    #     torch_dtype=torch.float16,
    # )
    # llm = LLM(model="/mnt/workspace/workgroup_dev/zhiqiang/MetaMath/trained_models/mistral_metamath", trust_remote_code=True, tensor_parallel_size=1) #tensor_parallel_size=1 meta-llama/Llama-2-70b-hf 01-ai/Yi-34B SeaLLMs/SeaLLM-Chat-13b meta-math/MetaMath-Mistral-7B

    # run evaluation
    # outputs = model.generate(test_data, sampling_params)
    generations = process_outputs(outputs, test_data, args.dataset_name)
    # print(f"The test accuracy on {args.dataset_name} is: {accuracy}")
    # logger.info(f'The test accuracy on {args.dataset_name} is: {accuracy}')
    weights = args.model_weights.split('/')[-2]
    writer(generations,
           f"/root/autodl-tmp/LLM-Adapters/generation_results/{args.base_model.replace('/root/autodl-tmp/', '')}/{weights}_{args.dataset_name}_generations.json")
    # writer(accuracy,
    #        f"/root/autodl-tmp/LLM-Adapters/math_evaluation/{args.base_model.replace('/root/autodl-tmp/', '')}/{weights}_{args.dataset_name}_accuracy.json")


if __name__ == '__main__':
    main()
