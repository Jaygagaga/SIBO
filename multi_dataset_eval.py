from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess

def evaluate(dataset, gpu, embedding_lambda):
    print('*******dataset:', dataset)

    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate.py \
               --model BLOOM-7B \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model '/root/autodl-tmp/bloomz-7b1' \
               --lora_weights './trained_models/llama-lora' \
                --embedding_lambda {embedding_lambda} \
                 --lora_weights '/root/autodl-tmp/LLM-Adapters/checkpoints/bloomz_7b1_bottleneck_att/math_10k/16_3e-4_3_01/' \ "

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu


datasets = ['AQuA', 'gsm8k', 'SVAMP', 'mawps']
gpus = [1, 2, 3]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))






