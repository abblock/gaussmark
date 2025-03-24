import datasets

import os
import json

import time
import torch
import hydra
from omegaconf import OmegaConf
from utils import hf_login, get_tokenizer


import transformers

from generate_text import get_prompts
from kgw_generate import generate_kgw, add_pvalues

from corrupt_detect_watermarks import corrupt_text






@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):


    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    

    hf_login(cfg)




    data_loading_start = time.time()
    prompts = get_prompts(cfg)
    data_loading_end = time.time()
    print(f"\nData loading ({len(prompts)} prompts) took {data_loading_end - data_loading_start:.0f} seconds\n")



    model_loading_start = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.name).to(cfg.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")


    generation_start = time.time()
    generations = generate_kgw(cfg, prompts, tokenizer, model)
    generation_end = time.time()
    print(f"\nGeneration took {generation_end - generation_start:.0f} seconds\n")


    corruption_start = time.time()
    tokenizer = get_tokenizer(cfg)
    outputs = corrupt_text(cfg, generations, tokenizer)
    corruption_end = time.time()
    print(f"\nCorruption took {corruption_end - corruption_start:.0f} seconds\n")




    detection_start = time.time()
    outputs = add_pvalues(outputs, model, tokenizer, cfg)
    detection_end = time.time()
    print(f"\nDetection took {detection_end - detection_start:.0f} seconds\n")


    pvalues = [output['pvalue'] for output in outputs if 'pvalue' in output]
    pvalue_avg = sum(pvalues) / len(pvalues)
    print(f"Average p-value: {pvalue_avg:.4f}")



    out_path = os.path.join(cfg.master_parent, "generations.json")
    print(f"Saving generations to {out_path}")
    with open(out_path, "w") as f:
        json.dump(outputs, f)

    times = {
        'generation': generation_end - generation_start,
        'detection': detection_end - detection_start,
    }
    out_path = os.path.join(cfg.master_parent, "times.json")
    print(f"Saving times to {out_path}")
    with open(out_path, "w") as f:
        json.dump(times, f)

if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Master time: {master_end - master_start:.0f} seconds")