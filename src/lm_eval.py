import hydra
from omegaconf import OmegaConf
import time

import shutil
import os
from utils import get_watermark_path, hf_login

from hf_core import VanillaLMWatermarker, LowRankLMWatermarker

import torch
from generate_text import get_model_type, get_watermark_param_names, _is_phi, _is_llama
import gc


def get_tasks(task_list, is_llama=False):
    out = ""
    for task in task_list:
        out += task + ","
    return out[:-1]

def get_lm_evaluation_harness():
    if 'lm-evaluation-harness' in os.listdir():
        print("lm-evaluation-harness already exists")
    else:
        command = """git clone https://github.com/EleutherAI/lm-evaluation-harness"""
        os.system(command)
        command = """pip install -e lm-evaluation-harness"""
        os.system(command)
        print("Cloned and installed lm-evaluation-harness")

def rename_results(output_path):
    """
    Renaming the results.json file to be in the output directory
    """
    results_path = os.path.join(output_path, os.listdir(output_path)[0])
    results_path = os.path.join(results_path, os.listdir(results_path)[0])
    os.system(f"mv {results_path} {output_path}/results.json")
    shutil.rmtree(os.path.dirname(results_path))

PHI_SYSTEM_INSTRUCTION = """
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    Answer the following question as best as you are able.  The question is:
    """




@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    hf_login(cfg)
    is_phi = _is_phi(cfg.model.name)
    is_llama = _is_llama(cfg.model.name)
    cfg = cfg.lm_eval
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    
    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)

    get_lm_evaluation_harness()

    task_string = get_tasks(cfg.tasks, is_llama=is_llama) # Change gsm8k to gsm8k_llama task.

    output_path = os.path.join(cfg.master_parent, cfg.output_path)

    if cfg.model_path is not None:
        model_path = cfg.model_path
    else: 
        model_path = os.path.join(get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop), 'watermarked-model')
    if cfg.is_unwatermarked_model:
        model_path = cfg.model.name

    if not os.path.exists(model_path) and not cfg.is_unwatermarked_model:
        print(f"Watermarked Model not found at {model_path}")
        return


    

    os.makedirs(output_path, exist_ok=True)
    
    if cfg.engine == "vllm":
        command = f"""
        lm_eval --model vllm \
            --model_args pretrained={model_path},max_model_len={cfg.max_model_len},tokenizer={cfg.tokenizer},gpu_memory_utilization={cfg.gpu_memory_utilization},max_num_seqs={cfg.max_num_seqs},enforce_eager={cfg.enforce_eager} \
            --tasks {task_string} \
            --batch_size auto \
            --trust_remote_code \
            --output_path {output_path} \
            --seed {cfg.seed},{cfg.seed+1},{cfg.seed+2},{cfg.seed+3} \
            --max_batch_size {cfg.max_batch_size}
            """

    elif cfg.engine == "hf":
        command = f"""
        lm_eval --model hf \
            --model_args pretrained={model_path},tokenizer={cfg.tokenizer} \
            --tasks {task_string} \
            --trust_remote_code \
            --output_path {output_path} \
            --seed {cfg.seed},{cfg.seed+1},{cfg.seed+2},{cfg.seed+3} \
            --batch_size {cfg.max_batch_size}
            """

    if is_phi:
        command += f" --system_instruction {PHI_SYSTEM_INSTRUCTION}"

    if not cfg.limit is None:
        if cfg.limit > 0:
            command += f" --limit {cfg.limit}"


    if cfg.track_memory_usage:
        device = torch.device(cfg.device)
        torch.cuda.reset_peak_memory_stats(device)

    gc.collect()
    torch.cuda.empty_cache()

    os.system(command)

    if cfg.track_memory_usage:
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1000
        max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1000
        print(f"Max memory allocated: {max_memory_allocated:.2f} GB")
        print(f"Max memory reserved: {max_memory_reserved:.2f} GB")


    try:
        rename_results(output_path)
    except NotADirectoryError:
        print("results.json already exists as a file.")





if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    