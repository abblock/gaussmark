import torch
import os

from huggingface_hub import login
import transformers




def get_token_level_logits(model, dataset, prompt_length=1):
    """
    Returns token level logits of a dataset according to a given model
    """
    return torch.nn.functional.log_softmax(model(dataset).logits, 2).gather(2, dataset.roll(-1).unsqueeze(2)).squeeze(2)[:,prompt_length:-1]





def get_sequence_level_logits(model, dataset, prompt_length=1):
    """
    Returns sequence level logits of a dataset according to a given model
    """
    return get_token_level_logits(model, dataset, prompt_length).sum(1)



def get_watermark_path(model_name, seed, parent='../models', rank=None):
    """
    Returns the path to the watermark model.  Assumes rank is none, in which case the VanillaWatermarker is assumed.  If rank is not none, the path to a LowRankWatermarker is returned
    """
    model_name = model_name.replace('/', '-')
    if rank is not None:
        return os.path.join(parent, f'{model_name}_seed_{seed}_rank_{rank}')
    else:
        return os.path.join(parent, f'{model_name}_seed_{seed}')


def hf_login(cfg):
    """
    Loads huggingface token from file and logs in
    """
    if os.path.exists(cfg.hf_token_path):
        with open(cfg.hf_token_path, 'r') as f:
            hf_token = f.read()
        
        login(token=hf_token, add_to_git_credential=False)
        del hf_token
        print('Logged in to Huggingface!')
    else:
        print('No Huggingface token found.  Please log in manually.')


def get_tokenizer(cfg):
    """
    Given a config, returns the appropriate tokenizer.
    """
    return transformers.AutoTokenizer.from_pretrained(cfg.model.name)