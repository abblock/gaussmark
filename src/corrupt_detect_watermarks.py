import torch
import json
import os
import time


import hydra 
from omegaconf import OmegaConf
from utils import get_watermark_path, hf_login, get_tokenizer
import shutil

import numpy as np
import transformers

from tqdm import tqdm
from hf_core import VanillaLMWatermarker, LowRankLMWatermarker

from detect_watermarks import load_text, add_pvalues

from functools import reduce

from transformers import MarianMTModel, MarianTokenizer



def _sanitize_token_list(token_list):
    return [int(token) for token in token_list]



def _add_tokens_at_indices(token_list, add_tokens, indices):
    token_list = _sanitize_token_list(token_list)
    out = []
    counter = 0
    for i in range(len(token_list)):
        if i in indices:
            out.append(add_tokens[counter])
            counter += 1
        out.append(token_list[i])

    return out

def _add_tokens_at_start(token_list, add_tokens):
    token_list = _sanitize_token_list(token_list)
    return add_tokens + token_list

def _add_tokens_at_end(token_list, add_tokens):
    token_list = _sanitize_token_list(token_list)
    return token_list + add_tokens



def add_random_tokens(token_list, token_frac, tokenizer, restrict_to_present_tokens=False, location='random', **kwargs):
    """
    Adds a random subset of size `token_frac * len(token_list)` to `token_list`. If `restrict_to_present_tokens` is True, the added tokens are resampled from tokens in current list. If location is start or end, tokens are added there.  Default is random indices.
    """
    num_tokens = int(token_frac * len(token_list))
    if restrict_to_present_tokens:
        add_tokens = [int(x) for x in np.random.choice(token_list, num_tokens, replace=True)]
    else:
        add_tokens = [int(x) for x in np.random.choice(list(range(tokenizer.vocab_size)), num_tokens)]
    
    if location == 'random':
        idxs = list(np.random.choice(len(token_list), num_tokens))
        return _add_tokens_at_indices(token_list, add_tokens, idxs)
    elif location == 'start':
        return _add_tokens_at_start(token_list, add_tokens)
    elif location == 'end':
        return _add_tokens_at_end(token_list, add_tokens)
    else:
        raise ValueError('location must be one of "random", "start", or "end"')



def substitute_random_tokens(token_list, token_frac, tokenizer, restrict_to_present_tokens=False, location='random', **kwargs):
    """
    Substitutes a random subset of size `num_tokens` in `token_list`.  If `restrict_to_present_tokens` is True, the substituted tokens are resampled from tokens in current list.
    """
    num_tokens = int(token_frac * len(token_list))
    token_list = _sanitize_token_list(token_list)
    if restrict_to_present_tokens:
        substitute_tokens = [int(x) for x in np.random.choice(token_list, num_tokens, replace=True)]
    else:
        substitute_tokens = [int(x) for x in np.random.choice(list(range(tokenizer.vocab_size)), num_tokens)]
    if location == 'random':
        idxs = list(np.random.choice(len(token_list), num_tokens))
    elif location == 'start':
        idxs = list(range(num_tokens))
    elif location == 'end':
        idxs = list(range(len(token_list)-num_tokens, len(token_list)))
    
    output = []
    counter = 0
    for i in range(len(token_list)):
        if i in idxs:
            output.append(substitute_tokens[counter])
            counter += 1
        else:
            output.append(token_list[i])

    return output




def remove_random_tokens(token_list, token_frac, tokenizer, location='random', **kwargs):
    """
    Removes a random subset of size `num_tokens` from `token_list`.  If location is start or end, tokens are removed from there.  Default is random indices.
    """
    num_tokens = int(token_frac * len(token_list))
    token_list = _sanitize_token_list(token_list)
    if location == 'random':
        # idxs = list(np.random.choice(len(token_list), num_tokens))
        idxs = torch.randperm(len(token_list))[:num_tokens]
        return [token_list[i] for i in range(len(token_list)) if i not in idxs]
    elif location == 'start':
        return token_list[num_tokens:]
    elif location == 'end':
        return token_list[:-num_tokens]
    else:
        raise ValueError('location must be one of "random", "start", or "end"')


def make_roundtrip_translation_function(cfg, tokenizer):
    """
    Use MarianMTModel to make a roundtrip translation function.
    """
    language = cfg.corruptions[0].language
    if language == 'french': ## Adapted from https://github.com/jthickstun/watermark/blob/main/experiments/c4-experiment.py#L107
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name).to(cfg.device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name).to(cfg.device)
    else:
        raise ValueError(f"Language {language} not supported")
    
    def roundtrip_translation(token_list):
        text = tokenizer.decode(token_list, skip_special_tokens=True)
        
        en_ne_tokens = en_ne_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(cfg.device)
        en_ne_translation = en_ne_model.generate(**en_ne_tokens)
        en_ne_text = en_ne_tokenizer.batch_decode(en_ne_translation, skip_special_tokens=True)[0]
        ne_en_tokens = ne_en_tokenizer(en_ne_text, return_tensors='pt', padding=False, truncation=True).to(cfg.device)
        ne_en_translation = ne_en_model.generate(**ne_en_tokens)
        ne_en_text = ne_en_tokenizer.batch_decode(ne_en_translation, skip_special_tokens=True)[0]
        new_tokens = tokenizer(ne_en_text, return_tensors='pt', padding=False, truncation=True)
        return new_tokens.input_ids.cpu()[0,:].numpy().tolist()
    
    return roundtrip_translation


def get_corruption_function(cfg, tokenizer):
    """
    Given a config `cfg`, returns a function that corrupts text according to the corruption parameters in `cfg`.
    """
    aliases = [corruption.alias for corruption in cfg.corruptions]
    if 'roundtrip_translation' in aliases:
        assert aliases[0] == 'roundtrip_translation', "Roundtrip translation must be the only corruption"
        return make_roundtrip_translation_function(cfg, tokenizer)
    else:
        corruption_functions = []
        for corruption in cfg.corruptions:
            if corruption.alias == 'add_random_tokens':
                print(f"Adding {corruption.token_frac} random tokens")
                fxn = lambda x: add_random_tokens(x, corruption.token_frac, tokenizer, **corruption.kwargs)
                corruption_functions.append(fxn)

            elif corruption.alias == 'remove_random_tokens':
                print(f"Removing {corruption.token_frac} random tokens")
                fxn = lambda x: remove_random_tokens(x, corruption.token_frac, tokenizer, **corruption.kwargs)
                corruption_functions.append(fxn)
            
            elif corruption.alias == 'substitute_random_tokens':
                print(f"Substituting {corruption.token_frac} random tokens")
                fxn = lambda x: substitute_random_tokens(x, corruption.token_frac, tokenizer, **corruption.kwargs)
                corruption_functions.append(fxn)


        ## Return function composing all functions in the list
        return lambda x: reduce(lambda y, fxn: fxn(y), corruption_functions, x)



def corrupt_text(cfg, generations, tokenizer):
    """
    Given a list of dictionaries `generations`, corrupts the text according to the corruption parameters in `cfg`.  The dictionaries in `generations` must have keys 'prompt_token_ids' and 'response_token_ids', which are renamed and replaced by the corrupted version.
    """
    corrupt_generations = []

    corruption_function = get_corruption_function(cfg.corruption_robust, tokenizer)
    for generation in generations:
        try:
            generation['original_prompt_token_ids'] = generation['prompt_token_ids']
            generation['original_response_token_ids'] = generation['response_token_ids']
            generation['original_response'] = generation['response']

            generation['response_token_ids'] = corruption_function(generation['response_token_ids'])
            generation['response'] = tokenizer.decode(generation['response_token_ids'], skip_special_tokens=True)

            if len(generation['response']) > 0:
                corrupt_generations.append(generation)
        except:
            print(f"Error computing corruption for text:\t{generation['prompt'] + generation['response']}")

    del corruption_function
    torch.cuda.empty_cache()

    return corrupt_generations



@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    if cfg.model.watermark_variance is None or cfg.model.watermark_variance == 0:
        print("No watermark variance specified, skipping watermark detection")
        return

    

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
        
    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)
    
    hf_login(cfg)

    model_loading_start = time.time()
    if cfg.model.path is not None:
        model_path = cfg.model.path
    else:
        model_path = get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop)
    
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    print(f"Loading model from {model_path}")
    if cfg.model.rank_to_drop is None:
        model = VanillaLMWatermarker.load_pretrained(model_path, keep_base_model=True, keep_watermarked_model=False) # Only need base model for watermark detection
    else:
        model = LowRankLMWatermarker.load_pretrained(model_path, keep_base_model=True, keep_watermarked_model=False)
    model.to(cfg.device)
    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")




    data_loading_start = time.time()
    generations = load_text(cfg)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")

    corruption_start = time.time()
    tokenizer = get_tokenizer(cfg)
    corrupt_generations = corrupt_text(cfg, generations, tokenizer)
    corruption_end = time.time()
    print(f"\nCorruption took {corruption_end - corruption_start:.0f} seconds\n")


    watermarking_start = time.time()
    generations_with_pvalues = add_pvalues(corrupt_generations, model, verbose=cfg.watermarking.verbose, block_size=cfg.model.robust_block_size, num_samples=cfg.model.robust_num_samples)
    watermarking_end = time.time()
    print(f"\nWatermark detection took {watermarking_end - watermarking_start:.0f} seconds\n")

    for gen in generations_with_pvalues: ## Convert tensors and arrays to lists for json serialization
        for key, val in gen.items():
            if isinstance(val, torch.Tensor):
                gen[key] = val.cpu().numpy().tolist()
            elif isinstance(val, np.ndarray):
                gen[key] = val.tolist()


    if cfg.corrupt_generations_path is not None:
        temp_path = os.path.join(cfg.master_parent, cfg.corrupt_generations_path)
    else:
        temp_path = os.path.join(cfg.master_parent, 'corrupt_generations.json')
    
    with open(temp_path, 'w') as f:
        print(f"Saving generations to {temp_path}")
        json.dump(generations_with_pvalues, f)
    


    pvalues = []
    log_probs = []
    base_model_logprobs = []
    for generation in generations_with_pvalues:
        pvalues.append(generation['pvalue'])
        log_probs.append(generation['logprobs'])
        base_model_logprobs.append(generation['base_model_logprobs'])
    
    median_pval = torch.tensor(pvalues).median().item()
    median_log_prob = torch.tensor(log_probs).median().item()
    median_base_model_log_prob = torch.tensor(base_model_logprobs).median().item()

    print(f"Median p-value: {median_pval:.3f}")
    print(f"Median log-prob: {median_log_prob:.3f}")
    print(f"Median base model log-prob: {median_base_model_log_prob:.3f}")




if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    