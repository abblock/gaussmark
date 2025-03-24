import datasets

import os
import json

import time
import torch
import hydra
from omegaconf import OmegaConf
from utils import hf_login


import transformers

from generate_text import get_prompts


def _tokenize_prompts(prompt_batch, tokenizer, device):
    return tokenizer(prompt_batch, padding=True, truncation=True, return_tensors="pt").to(device)

def _parse_batch(outputs, prompt_batch, tokenizer, model, generation_config, watermarking_config, device):
    """
    Given a batch of prompts, generates using HF and then parses the outputs into a list of dictionaries appended to the outputs list.
    """
    tokenized_prompts = _tokenize_prompts(prompt_batch, tokenizer, device)
    model_outputs = model.generate(**tokenized_prompts, generation_config=generation_config, watermarking_config=watermarking_config)
    model_output_strings = tokenizer.batch_decode(model_outputs)

    for i in range(len(prompt_batch)):
        
        output = {}
        output['prompt'] = prompt_batch[i]
        output['prompt_token_ids'] = tokenized_prompts['input_ids'][i].cpu().tolist()
        output['response'] = model_output_strings[i]
        output['response_token_ids'] = model_outputs[i].cpu().tolist()
        
        outputs.append(output)


def get_kgw_config(cfg):
    if cfg.other_gen.do_watermarking:
        return transformers.WatermarkingConfig(
            greenlist_ratio=cfg.other_gen.greenlist_ratio,
            bias=cfg.other_gen.bias,
            seeding_scheme=cfg.other_gen.seeding_scheme,
            context_width=cfg.other_gen.context_width
            )
    elif cfg.other_gen.bias <= 0:
        return None
    else:
        return None

def get_generation_config(cfg):
    if cfg.sampling.top_k == -1:
        cfg.sampling.top_k = 0

    return transformers.GenerationConfig(
        do_sample=True,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        temperature=cfg.sampling.temperature,
        max_new_tokens=cfg.sampling.max_tokens
    )
  



def get_detector(model, device, watermarking_config):
    return transformers.WatermarkDetector(model_config=model.config, device=device, watermarking_config=watermarking_config)

def generate_kgw(cfg, prompts, tokenizer, model, detector=None):

    transformers.set_seed(cfg.seed)
    device = cfg.device
    batch_size = cfg.other_gen.batch_size
    watermarking_config = get_kgw_config(cfg)

    
    generation_config = get_generation_config(cfg)
    outputs = []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i + batch_size]
        _parse_batch(outputs, prompt_batch, tokenizer, model, generation_config, watermarking_config, device)



    
    return outputs



def add_pvalues(outputs, model,tokenizer, cfg):

    device = cfg.device
    batch_size = cfg.other_gen.batch_size
    watermarking_config = get_kgw_config(cfg)
    detector = get_detector(model, device, watermarking_config=watermarking_config)
    for i in range(0, len(outputs), batch_size):
        output_batch = outputs[i:i + batch_size]
        response_token_ids = [output['response_token_ids'] for output in output_batch]
        response_tokens = torch.tensor(response_token_ids).to(device)
        detection_out_watermarked = detector(response_tokens, return_dict=True)
        pvalues = detection_out_watermarked.p_value
        for j in range(len(output_batch)):
            outputs[i+j]['pvalue'] = pvalues[j]
    return outputs


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
    outputs = generate_kgw(cfg, prompts, tokenizer, model)
    generation_end = time.time()
    print(f"\nGeneration took {generation_end - generation_start:.0f} seconds\n")


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



