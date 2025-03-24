import torch
import copy
import numpy as np
import os
import json
import transformers
from vllm import LLM
from utils import get_sequence_level_logits




class BaseWatermark(torch.nn.Module):

    def __init__(
            self,
            base_model,
            watermark_param_names,
            path,
            loss='cross_entropy',
            variance=1e-5,
            seed = None,
            tokenizer_name=None,
            keep_base_model=True,
            keep_watermarked_model=True,
            gpu_memory_utilization=0.9,
            max_num_seqs=None
        ):
        """
        Base watermark class that other watermarkers inherit from
        """
        super(BaseWatermark, self).__init__()

        


        if keep_base_model or not os.path.exists(os.path.join(path, 'watermarked-model')):
            self.base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model)
            self.max_position_embedding = self.base_model.config.max_position_embeddings
            self.water_marked_model = copy.deepcopy(self.base_model)
        else:
            print("Dropping base model (generation mode)")
            self.base_model = None

        self.variance = variance
        self.std_dev = torch.sqrt(torch.tensor(variance)).item()

       
        self.watermark_param_names = watermark_param_names
        
        
        self.loss_name = loss

        if self.loss_name == 'cross_entropy':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Only cross_entropy loss is supported right now")


        if tokenizer_name is None:
            self.tokenizer_name = base_model
        else:
            self.tokenizer_name = tokenizer_name

        self.gpu_memory_utilization = gpu_memory_utilization



        if seed is None:
            seed = self.generate_seed()
        self.seed = seed
        
        if keep_watermarked_model and not os.path.exists(os.path.join(path, 'watermarked-model')):
            self.watermarks = self.watermark_parameters()

            if len(self.watermarks) == 0:
                raise ValueError(f"No watermark parameters found, make sure watermark parameters are valid.  Entered parameters: {watermark_param_names}")
            
            self.save_pretrained(path)            


        if not keep_base_model:
            print("Dropping base model (generation mode)")
            self.base_model = None


        if keep_watermarked_model:
            self.watermarked_model = LLM(os.path.join(path, 'watermarked-model'), tokenizer=self.tokenizer_name, gpu_memory_utilization=self.gpu_memory_utilization)
        else:
            print("Dropping watermarked model (detection mode)")
            self.watermarked_model = None
            if keep_base_model: # Only keep grads for watermark parameters
                for name, param in self.base_model.named_parameters():
                    if self.is_watermark_parameter(name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False



    def generate(self, *args, **kwargs):
        """
        Generates text from the watermarked model
        """
        return self.watermarked_model.generate(*args, **kwargs)



    def is_watermark_parameter(self, name):
        """
        Returns True if the parameter is a watermark parameter
        """
        return name in self.watermark_param_names

    def generate_seed(self):
        """
        Generates random seed for watermarking
        """
        return np.random.randint(0, 1e6)


    def reset(self, reset_watermarker=False, seed=None):
        """
        Resets the watermarker
        """
        raise NotImplementedError
    
    def update_variance(self, new_variance):
        """
        Updates the variance of the watermark
        """
        self.variance = new_variance
        self.std_dev = torch.sqrt(torch.tensor(new_variance)).item()
        self.watermark_parameters()


    def watermark_parameters(self):
        """
        Watermarks the model
        """
        raise NotImplementedError


    def get_watermark_score(self, prompt, output, add_to_history=True):
        """
        Given prompt and output, returns the watermark score.  Also, adds the score to the history if add_to_history is True
        """
        raise NotImplementedError


    def __str__(self):
        return self.base_model.__str__()



    def get_p_value(self, score, var_under_null):
        """
        Returns the p-value of the score
        """
        raise NotImplementedError


    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass of the watermarker
        """
        return self.water_marked_model(x)



    @torch.no_grad()
    def get_base_model_logits(self, text):
        """
        Returns the logits of the text
        """
        assert self.keep_base_model, "Base model is not kept in memory"

        logits = self.base_model(text).logits
        return logits[torch.arange(logits.shape[0]), torch.arange(logits.shape[1]), text]


    @staticmethod
    def _get_p_value(score, sample_scores):
        """
        Given a score and a sample of scores under the null hypothesis, returns the p-value, i.e, the fraction of the sample scores that are at least as large as the score
        """
        return 1.0 - (sample_scores >= score).float().mean()


    @staticmethod
    def _quantile_aggregation(scores, quantile=0.5):
        """
        Given a torch tensor of shape either (num_tokens,) or (num_samples, num_tokens), returns the quantile aggregation of the scores
        """
        if len(scores.shape) == 1:
            return scores.quantile(quantile)
        else:
            return scores.quantile(quantile, dim=1)



    def save_pretrained(self, path):
        """
        Saves the watermarked model
        """
        print(f"Saving watermarked model to {path}")
        os.makedirs(path, exist_ok=True)
        self.water_marked_model.save_pretrained(os.path.join(path, "watermarked-model"))
        self.base_model.save_pretrained(os.path.join(path, "base-model"))

        watermarks = {name: watermark.cpu().detach() for name, watermark in self.watermarks.items()}
        os.makedirs(os.path.join(path, 'watermarks'))
        for name, watermark in watermarks.items():
            torch.save(watermark, os.path.join(path, 'watermarks', f"{name}.pt"))

        params = {
            "watermark_param_names": self.watermark_param_names,
            "variance": self.variance,
            'loss_name': self.loss_name,
            'seed': self.seed,
            'tokenizer_name': self.tokenizer_name
        }
        if hasattr(self, 'rank_to_drop'):
            params['rank_to_drop'] = self.rank_to_drop
        else:
            params['rank_to_drop'] = 0
        
        if hasattr(self, 'laserized'):
            params['laserized'] = self.laserized
        else:
            params['laserized'] = False


        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(params, f)

        if hasattr(self, 'projectors'):
            os.makedirs(os.path.join(path, 'projectors'))
            for name, projector in self.projectors.items():
                torch.save(projector.cpu().detach(), os.path.join(path, 'projectors', f"{name}.pt"))
    
    
    def to(self, device):
        """
        Moves the model to the device
        """
        if self.base_model is not None:
            self.base_model.to(device)
        if self.watermarked_model is not None:
            self.water_marked_model.to(device)
        
        for name, watermark in self.watermarks.items():
            self.watermarks[name] = watermark.to(device)
        
        if hasattr(self, 'low_rank_approximations'):
            for name, low_rank_approximation in self.low_rank_approximations.items():
                self.low_rank_approximations[name] = low_rank_approximation.to(device)
        
        if hasattr(self, 'projectors'):
            for name, projector in self.projectors.items():
                self.projectors[name] = projector.to(device)
        


        self.device = device
        return self

    def get_score_and_pvalue_cumnormal(self, prompt, output):
        raise NotImplementedError
    

    def get_robust_score_and_pvalue(self, prompt, output, block_size=50, num_samples=10):
        """
        Robustifies the watermark detection test by taking a maximum over blocks of size `block_size` of the test statistic.  Returns the bonferroni corrected score.
        """
        if block_size is None:
            return self.get_score_and_pvalue_cumnormal(prompt, output)
        prompt_len = len(prompt)
        response_len = len(output)
        
        if block_size >= response_len + min(prompt_len, block_size) - block_size: ## If block size is too big, just return the nonrobust score and p-value
            return self.get_score_and_pvalue_cumnormal(prompt, output)

        num_samples = min(num_samples, response_len - block_size + min(prompt_len, block_size))

        initial_idxs = np.random.choice(np.arange(-min(prompt_len, block_size), response_len - block_size), num_samples, replace=False)
        new_inputs = []
        for idx in initial_idxs:
            
            if idx < 0:
                new_prompt = prompt[-idx:]
            else:
                new_prompt = []
            new_output = output[max(0, idx):idx+block_size]
            new_inputs.append((new_prompt, new_output))


        scores, pvalues, vars_under_null, logprobs = [], [], [], []
        for new_prompt, new_output in new_inputs:
            score, pvalue, var_under_null, logprob = self.get_score_and_pvalue_cumnormal(new_prompt, new_output)
            scores.append(score)
            pvalues.append(pvalue)
            vars_under_null.append(var_under_null)
            logprobs.append(logprob)
        
        best_idx = np.argmin(pvalues)
        return scores[best_idx], num_samples * pvalues[best_idx], vars_under_null[best_idx], logprobs[best_idx]
        











class VanillaLMWatermarker(BaseWatermark):

    def __init__(
            self,
            base_model,
            watermark_param_names,
            path,
            loss,
            variance=1e-5,
            seed = None,
            tokenizer_name=None,
            keep_base_model=True,
            keep_watermarked_model=True,
            gpu_memory_utilization=0.9,
            max_num_seqs=None
            ):
        """
        Initializes the watermarker

        Args:
        
        base_model: string or transformers.AutoModelForCausalLM
            The model to be watermarked
        watermark_param_names: list
            The names of the parameters to be watermarked
        path: str
            The path to save the watermarked model
        loss: string
            The loss function to be used to detect the watermarks.  Right now only `cross_entropy` is supported
        variance: float (default=1e-5)
            The variance of the watermark. 
        seed: int (default=None)
            The seed for the random number generator to be used to generate watermark.
        tokenizer_name: str (default=None)
            The name of the tokenizer to be used.  If None, the tokenizer of the base model is used.
        keep_base_model: bool (default=True)
            If True, the base model is kept in memory.  If False, the base model is deleted after the watermarked model is created.  Set to False for text generation, only needs to be true for watermarking.
        """
        super(VanillaLMWatermarker, self).__init__(
            base_model,
            watermark_param_names,
            path,
            loss,
            variance,
            seed,
            tokenizer_name,
            keep_base_model,
            keep_watermarked_model,
            gpu_memory_utilization,
            max_num_seqs
        )
        

    

    def reset(self, reset_watermarker=False, seed=None):
        """
        Resets the watermarker
        """
        if reset_watermarker: ## Resets watermarker as well as scores and variances
            if seed is None:
                seed = self.generate_seed()
            self.seed = seed
            self.watermark_parameters()

    

    def watermark_parameters(self):
        """
        Watermarks the model
        """
    
        torch.manual_seed(self.seed)

        watermarks = {}

        for (name, param), base_model_param in zip(self.water_marked_model.named_parameters(), self.base_model.parameters()):
            if self.is_watermark_parameter(name):
                watermark = torch.randn_like(param) * self.std_dev
                watermarks[name] = watermark
                param.data = base_model_param.data +  watermark
                base_model_param.requires_grad = True
            else:
                base_model_param.requires_grad = False

        self.watermarks = watermarks
        return watermarks
        



    def _get_summed_gradient(self, prompt, output):
        """
        Returns gradient of the loss evaluated on the output of the base model given the prompt with respect to the parameters.  Assumes prompt is list of token_ids and output is list of token_ids
        """
        prompt_tensor = torch.tensor(prompt).unsqueeze(0).to(self.base_model.device).long()
        output_tensor = torch.tensor(output).unsqueeze(0).to(self.base_model.device).long()
        prompt_len = prompt_tensor.shape[1]

        inputs = torch.cat([prompt_tensor, output_tensor], dim=1)

        self.base_model.zero_grad()
        cumulative_logits = get_sequence_level_logits(self.base_model, inputs, prompt_length=prompt_len)
        cumulative_logits.backward()
        return {name: param.grad for name, param in self.base_model.named_parameters() if self.is_watermark_parameter(name)}, cumulative_logits.item()

    @torch.no_grad()
    @staticmethod
    def _get_watermark_score(gradient_dict, watermark_dict):
        """
        Given the gradient dictionary, returns the watermark score as the inner product between the gradient and the watermark.
        """
        score = 0.0
        for name, grad in gradient_dict.items():
            if name in watermark_dict.keys():
                score += torch.inner(grad.flatten(), watermark_dict[name].flatten())
        
        return score.item()
    

    @torch.no_grad()
    def get_variance_under_null(self, gradient_dict):

        norm_square = 0.0
        for _, grad in gradient_dict.items():
            norm_square += grad.norm().square().item()

        return self.variance * norm_square
    

    @staticmethod
    def inverse_gaussian_cdf(statistic, var, mean=0.0):
        """
        Returns the p-value of the statistic under the null hypothesis that the watermark is not present
        """
        std_dev = torch.sqrt(torch.tensor(var)).item()
        normalized_statistic = (statistic - mean) / std_dev
        return (1.0 -  torch.distributions.normal.Normal(0.0,1.0).cdf(torch.tensor(normalized_statistic))).item()




    def get_score_and_pvalue_cumnormal(self, prompt, output):
        """
        Computes the score of the test statistic (summed gradients over time steps) and the p-value. (Non-robust test)
        Returns the test statistic, the p_value under the null, the variance of the Gaussian under the null, and the base_model's log probabilities of the sequence
        """
        try:
            gradients, log_prob = self._get_summed_gradient(prompt, output)
            score = VanillaLMWatermarker._get_watermark_score(gradients, self.watermarks)
            var_under_null = self.get_variance_under_null(gradients)
            p_value = VanillaLMWatermarker.inverse_gaussian_cdf(score, var_under_null)
            return score, p_value, var_under_null, log_prob
        except Exception as e:
            print(e)
            return 0.0, 1.0, 0.0, 0.0






    @classmethod
    def load_pretrained(cls, path, keep_base_model=True, keep_watermarked_model=True,gpu_memory_utilization=0.9, max_num_seqs=None):
        """
        Loads the watermarked model
        """

        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.load(f)
        
        model_path = os.path.join(path, "base-model")

        watermarker = cls(
            model_path,
            params["watermark_param_names"],
            path=path,
            loss=params['loss_name'],
            seed=params['seed'],
            variance=params["variance"],
            tokenizer_name=params['tokenizer_name'],
            keep_base_model=keep_base_model,
            keep_watermarked_model=keep_watermarked_model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs
            )
    

        
        watermarks = {}
        for name in params["watermark_param_names"]:
            watermarks[name] = torch.load(os.path.join(path, 'watermarks', f"{name}.pt"))
        watermarker.watermarks = watermarks
        return watermarker


















class LowRankLMWatermarker(BaseWatermark):

    def __init__(
            self,
            base_model,
            watermark_param_names,
            path,
            loss,
            variance=1e-5,
            seed = None,
            tokenizer_name=None,
            keep_base_model=True,
            keep_watermarked_model=True,
            rank_to_drop=4,
            gpu_memory_utilization=0.9,
            max_num_seqs=None
            ):
        """
        LowRank watermarker, where gaussian noise is only added to the orthogonal complement of the low rank approximation of the weight matrix.

        Args:
        
        base_model: string or transformers.AutoModelForCausalLM
            The model to be watermarked
        watermark_param_names: list
            The names of the parameters to be watermarked
        path: str
            The path to save the watermarked model
        loss: string
            The loss function to be used to detect the watermarks.  Right now only `cross_entropy` is supported
        variance: float (default=1e-5)
            The variance of the watermark. 
        seed: int (default=None)
            The seed for the random number generator to be used to generate watermark.
        tokenizer_name: str (default=None)
            The name of the tokenizer to be used.  If None, the tokenizer of the base model is used.
        keep_base_model: bool (default=True)
            If True, the base model is kept in memory.  If False, the base model is deleted after the watermarked model is created.  Set to False for text generation, only needs to be true for watermarking.
        rank_to_drop: int (default=4)
            The rank of the low rank approximation of the weight onto whose orthogonal complement we project the watermark.
        """
        
        
        self.rank_to_drop = rank_to_drop
        
        super(LowRankLMWatermarker, self).__init__(
            base_model,
            watermark_param_names,
            path,
            loss,
            variance,
            seed,
            tokenizer_name,
            keep_base_model,
            keep_watermarked_model,
            gpu_memory_utilization,
            max_num_seqs
        )

        
        

    def reset(self, reset_watermarker=False, seed=None):
        """
        Resets the watermarker
        """
        if reset_watermarker: ## Resets watermarker as well as scores and variances
            if seed is None:
                seed = self.generate_seed()
            self.seed = seed
            self.watermark_parameters()

   
    @torch.no_grad()
    @staticmethod
    def _get_low_rank_watermark(weight, rank_to_drop, niter=4):
        """
        Get the low rank watermark of a weight matrix.
        """

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        U_k = U[:, :rank_to_drop]
        projector = torch.eye(U_k.shape[0]) - U_k @ U_k.T

        raw_watermark = torch.randn_like(weight)
        watermark = projector @ raw_watermark
        return watermark, projector



    def watermark_parameters(self):
        """
        Watermarks the model
        """
    
        torch.manual_seed(self.seed)

        watermarks = {}
        projectors = {}

        for (name, param), base_model_param in zip(self.water_marked_model.named_parameters(), self.base_model.parameters()):
            if self.is_watermark_parameter(name):
                watermark, projector = LowRankLMWatermarker._get_low_rank_watermark(base_model_param.data, self.rank_to_drop)
                watermark = self.std_dev * watermark
                watermarks[name] = watermark
                projectors[name] = projector
                param.data = base_model_param.data +  watermark
                base_model_param.requires_grad = True
            else:
                base_model_param.requires_grad = False

        self.watermarks = watermarks
        self.projectors = projectors
        return watermarks
        
    def make_low_rank_projectors(self):
        """
        Returns the projectors to the low rank approximations of the watermark parameters
        """
        projectors = {}
        for name, param in self.base_model.named_parameters():
            if self.is_watermark_parameter(name):
                _, projector = LowRankLMWatermarker._get_low_rank_watermark(param.data, self.rank_to_drop)
                projectors[name] = projector
        
        self.projectors = projectors
        return projectors


    def _get_summed_gradient(self, prompt, output):
        """
        Returns gradient of the loss evaluated on the output of the base model given the prompt with respect to the parameters.  Assumes prompt is list of token_ids and output is list of token_ids
        """
        prompt_tensor = torch.tensor(prompt).unsqueeze(0).to(self.base_model.device).long()
        output_tensor = torch.tensor(output).unsqueeze(0).to(self.base_model.device).long()
        prompt_len = prompt_tensor.shape[1]

        inputs = torch.cat([prompt_tensor, output_tensor], dim=1)

        self.base_model.zero_grad()
        cumulative_logits = get_sequence_level_logits(self.base_model, inputs, prompt_length=prompt_len)

        


        cumulative_logits.backward()

        gradients = {}
        for name, param in self.base_model.named_parameters():
            if self.is_watermark_parameter(name):
                gradient = param.grad
                projected_gradient = self.projectors[name] @ gradient
                gradients[name] = projected_gradient
        
        
        return gradients, cumulative_logits.item()

    @torch.no_grad()
    @staticmethod
    def _get_watermark_score(gradient_dict, watermark_dict):
        """
        Given the gradient dictionary, returns the watermark score as the inner product between the gradient and the watermark.
        """
        score = 0.0
        for name, grad in gradient_dict.items():
            if name in watermark_dict.keys():
                score += torch.inner(grad.flatten(), watermark_dict[name].flatten())
        
        return score.item()
    

    @torch.no_grad()
    def get_variance_under_null(self, gradient_dict):

        norm_square = 0.0
        for _, grad in gradient_dict.items():
            norm_square += grad.norm().square().item()

        return self.variance * norm_square
    

    @staticmethod
    def inverse_gaussian_cdf(statistic, var, mean=0.0):
        """
        Returns the p-value of the statistic under the null hypothesis that the watermark is not present
        """
        std_dev = torch.sqrt(torch.tensor(var)).item()
        normalized_statistic = (statistic - mean) / std_dev
        return (1.0 -  torch.distributions.normal.Normal(0.0,1.0).cdf(torch.tensor(normalized_statistic))).item()




    def get_score_and_pvalue_cumnormal(self, prompt, output):
        """
        Computes the score of the test statistic (summed gradients over time steps) and the p-value. (Non-robust test)
        Returns the test statistic, the p_value under the null, the variance of the Gaussian under the null, and the base_model's log probabilities of the sequence
        """
        try:
            gradients, log_prob = self._get_summed_gradient(prompt, output)
            score = LowRankLMWatermarker._get_watermark_score(gradients, self.watermarks)
            var_under_null = self.get_variance_under_null(gradients)
            p_value = LowRankLMWatermarker.inverse_gaussian_cdf(score, var_under_null)
            return score, p_value, var_under_null, log_prob
        except Exception as e:
            print(e)
            return 0.0, 1.0, 0.0, 0.0







    @classmethod
    def load_pretrained(cls, path, keep_base_model=True, keep_watermarked_model=True, gpu_memory_utilization=0.9, max_num_seqs=None):
        """
        Loads the watermarked model
        """

        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.load(f)
        
        model_path = os.path.join(path, "base-model")

        watermarker = cls(
            model_path,
            params["watermark_param_names"],
            path=path,
            loss=params['loss_name'],
            seed=params['seed'],
            variance=params["variance"],
            tokenizer_name=params['tokenizer_name'],
            keep_base_model=keep_base_model,
            keep_watermarked_model=keep_watermarked_model,
            rank_to_drop=params['rank_to_drop'],
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs
            )
    
        if os.path.exists(os.path.join(path, 'projectors')):
            projectors = {}
            for name in params["watermark_param_names"]:
                projectors[name] = torch.load(os.path.join(path, 'projectors', f"{name}.pt"))
            watermarker.projectors = projectors
        else:
            watermarker.projectors = watermarker.make_low_rank_projectors()


        watermarks = {}
        for name in params["watermark_param_names"]:
            watermarks[name] = torch.load(os.path.join(path, 'watermarks', f"{name}.pt"))
        watermarker.watermarks = watermarks
        return watermarker


