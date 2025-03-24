import torch
from vllm import LLM, SamplingParams
import transformers
import datasets

import os
import json

import time

import hydra
from omegaconf import OmegaConf


from utils import get_watermark_path, hf_login
from hf_core import VanillaLMWatermarker, LowRankLMWatermarker





def _is_llama(model_name):
    """
    Returns whether the model is a llama model
    """
    return 'meta-llama' in model_name

def _is_mistral(model_name):
    """
    Returns whether the model is a mistral model
    """
    return 'mistralai' in model_name

def _is_phi(model_name):
    """
    Returns whether the model is a phi model
    """
    return 'microsoft/Phi' in model_name




def get_data(cfg):
    """
    Downloads dataset from HF and loads list of prompts.
    """
    data = datasets.load_dataset(cfg.data.name, cfg.data.subdata_name, split=cfg.data.split, streaming=True)
    prompts = []
    counter = 0
    for datum in iter(data):
        prompt = datum[cfg.data.prompt_field_name]
        prompts.append(prompt)
        counter += 1
        if counter == cfg.data.max_samples:
            break

    return prompts


def _truncate_prompt(prompt, max_length):
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    return prompt

def _randomly_truncate_prompt(prompt,max_length, min_frac, max_frac=1.0):

    frac = torch.rand(1).clip(min=min_frac, max=max_frac).item()
    length = int(frac * max_length)
    return _truncate_prompt(prompt, length)

def _phi3_ify(prompt):
    """
    Given a prompt, returns the phi3-ified prompt
    """
    prefix = """
    <|system|>
    You are a helpful assistant.<|end|>
    <|user|>
    Complete the following prompt:<|end|>
    <|assistant|>\n
    """
    return prefix + prompt


def phi3ify_prompts(prompts):
    """
    Given a list of prompts, returns a list of phi3-ified prompts
    """
    return [_phi3_ify(prompt) for prompt in prompts]

def get_prompts(cfg):

    raw_prompts = get_data(cfg)
    if cfg.data.truncation_type == 'fixed':
        truncate = lambda prompt:_truncate_prompt(prompt, cfg.data.max_prompt_length)
    elif cfg.data.truncation_type == 'random':
        truncate = lambda prompt:_randomly_truncate_prompt(prompt, cfg.data.max_prompt_length, cfg.data.min_frac, cfg.data.max_frac)
    
    prompts = [truncate(prompt) for prompt in raw_prompts]

    if _is_phi(cfg.model.name):
        prompts = phi3ify_prompts(prompts)

    return prompts



def get_model_type(model_name):
    """
    Returns the type of the model given the model name
    """
    if _is_llama(model_name):
        return 'llama'
    elif _is_mistral(model_name):
        return 'mistral'
    elif _is_phi(model_name):
        return 'phi'
    else:
        return None 


def _get_llama_param_name(layer_idx, proj_type, param_type='weight'):
    """
    Returns the name of the llama parameter given the layer index and the parameter type. If layer index is None or negative, returns the lm_head's weight.
    """
    if layer_idx is None or int(layer_idx) < 0: # Return lm parameter
        name = f'lm_head.weight'
    else: # Return mlp parameters, earlier in layer
        acceptable_proj_types = {'gate_proj', 'up_proj', 'down_proj', 'gate_up_proj'}
        acceptable_param_types = {'weight', 'bias'}
        assert proj_type in acceptable_proj_types, f"proj_type must be one of {acceptable_proj_types}"
        assert param_type in acceptable_param_types, f"param_type must be one of {acceptable_param_types}"
        name = f'model.layers.{layer_idx}.mlp.{proj_type}.{param_type}'
    return name

def _get_mistral_param_name(layer_idx, proj_type, param_type='weight'):
    """
    Returns the name of the mistral parameter given the layer index and the parameter type. If layer index is None or negative, returns the lm_head's weight.
    """
    if layer_idx is None or int(layer_idx) < 0: # Return lm parameter
        name = f'lm_head.weight'
    else: # Return mlp parameters, earlier in layer
        acceptable_proj_types = {'gate_proj', 'up_proj', 'down_proj'}
        acceptable_param_types = {'weight', 'bias'}
        assert proj_type in acceptable_proj_types, f"proj_type must be one of {acceptable_proj_types}"
        assert param_type in acceptable_param_types, f"param_type must be one of {acceptable_param_types}"
        name = f'model.layers.{layer_idx}.mlp.{proj_type}.{param_type}'
    return name



def get_watermark_param_names(cfg):
    """
    Given the config with cfg.model.watermakr_param_names in `layer_idx@@@proj_type@@@param_type` format, returns the parameter names for the llama model.
    If layer_ids is negative, returns the lm_head.weight.  proj_type can be one of {'gate_proj', 'up_proj', 'down_proj'}. param_type can be one of {'weight', 'bias'}.
    """
    model_type = get_model_type(cfg.model.name)

    if model_type == 'llama':
        _get_param_name = _get_llama_param_name
    elif model_type == 'mistral':
        _get_param_name = _get_mistral_param_name
    elif model_type == 'phi':
        _get_param_name = _get_llama_param_name
    else:
        raise ValueError(f"Model {cfg.model.name} not recognized type")

    raw_param_names = cfg.model.watermark_param_names

    param_names =[]
    for raw_param_name in raw_param_names:
        layer_idx, proj_type, param_type = raw_param_name.split('@@@')
        param_name = _get_param_name(layer_idx, proj_type, param_type)
        param_names.append(param_name)
    
    return param_names


def get_sampling_params(cfg, save_all_responses=True):
    """
    Returns vllm sampling parameters from hydra config
    """

    return SamplingParams(
        n = 1,
        best_of=1,
        top_p = cfg.sampling.top_p,
        seed = cfg.sampling.seed,
        max_tokens=cfg.sampling.max_tokens,
        temperature=cfg.sampling.temperature,
        logprobs=0, # Return logprobs
        top_k=cfg.sampling.top_k,
    )


def _parse_response(request_output):
    """
    Given a vllm request output, returns a parsed dictionary containing the necessary fields
    """
    output = {}

    output['prompt'] = request_output.prompt
    output['prompt_token_ids'] = list(request_output.prompt_token_ids)
    response = request_output.outputs[0] # take most likely response
    
    output['response'] = str(response.text)
    output['response_token_ids'] = list(response.token_ids)
    output['logprobs'] = float(response.cumulative_logprob) # Takes sequence level logprobs of response

    return output


def parse_responses(request_outputs):
    """
    Given a list of vllm request outputs, returns a list of parsed dictionaries
    """
    return [_parse_response(request_output) for request_output in request_outputs]












@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    


    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    
    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)

    hf_login(cfg) # Log in to HF if path to hf_token is in cfg.

    model_loading_start = time.time()
    if cfg.model.watermark_variance == 0:
        model = LLM(cfg.model.name, tokenizer=cfg.tokenizer.name)
    elif cfg.model.path is not None:
        model_path = cfg.model.path
    else:
        model_path = get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop)
    
        print(f"\nWatermarking model {cfg.model.name} with seed {cfg.model.seed}\n")


        model_type = get_model_type(cfg.model.name)

        if model_type is not None: # get watermark param names from config
            watermark_param_names = get_watermark_param_names(cfg)
        else:
            watermark_param_names = OmegaConf.to_object(cfg.model.watermark_param_names)
        print(f"\nWatermarking parameters: {watermark_param_names}\n")

        if cfg.model.rank_to_drop is None or cfg.model.rank_to_drop == 0:
            model = VanillaLMWatermarker(
                cfg.model.name,
                watermark_param_names=watermark_param_names,
                path = model_path,
                loss = cfg.model.watermark_loss,
                variance=cfg.model.watermark_variance,
                seed = cfg.seed,
                tokenizer_name=cfg.tokenizer.name,
                keep_base_model=False,
                keep_watermarked_model=True,
                gpu_memory_utilization=cfg.sampling.gpu_memory_utilization
            )
        else:
                model = LowRankLMWatermarker(
                cfg.model.name,
                watermark_param_names=watermark_param_names,
                path = model_path,
                loss = cfg.model.watermark_loss,
                variance=cfg.model.watermark_variance,
                seed = cfg.seed,
                tokenizer_name=cfg.tokenizer.name,
                keep_base_model=False,
                keep_watermarked_model=True,
                rank_to_drop=cfg.model.rank_to_drop,
                gpu_memory_utilization=cfg.sampling.gpu_memory_utilization
            )

    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")


    data_loading_start = time.time()
    prompts = get_prompts(cfg)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")

    sampling_params = get_sampling_params(cfg)


    generation_start = time.time()
    responses = model.generate(prompts, sampling_params)
    responses = parse_responses(responses)
    generation_end = time.time()
    print(f"\nGeneration took {generation_end - generation_start:.0f} seconds\n")
    
   
    responses_path = os.path.join(cfg.master_parent, 'generations.json')
    print(f"\nSaving responses to {responses_path}\n")
    with open(responses_path, 'w') as f:
        json.dump(responses, f)
    


    





if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    






