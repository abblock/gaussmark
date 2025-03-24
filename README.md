# Code for GaussMark: A Practical Approach for Structural Watermarking of Language Models

This is the code for [GaussMark: A Practical Approach for Structural Watermarking of Language Models](https://arxiv.org/abs/2501.13941) by Adam Block, Alexander Rakhlin, and Ayush Sekhari.

To run, first make a virtual environment and install the requirements with `pip install -r requirements.txt`.  All configs are input through [hydra](https://hydra.cc/docs/intro/).  In order to generate watermarked text, run:

```python
python src/generate_text.py model.name=<hf-path-to-model> model.watermark_param_names=[<layer>@@@<param>@@@weight] model.watermark_variance=<variance> model.rank_to_drop=<rank> data.name=<hf-path-to-data>
```

Some relevant parameters are:
- `model.name` is a Huggingface path to a model, e.g., `'microsoft/Phi-3-mini-4k-instruct'`
- `model.watermark_param_names` a list of strings detailing which parameters to watermark, specified by layer and what kind of parameter, e.g. `30@@@down_proj@@@weight`
- `model.watermark_variance` a float for the variance of the gaussian to be added, e.g. `1e-05`.  If unwatermarked text is desired, set this to 0.0.
- `model.rank_to_drop` optionally use the RankReduced version of Gaussmark, from the paper.  If set to `0` then this is ignored and the vanilla Gaussmark is used.
- `data.name` is a Huggingface path to a dataset, e.g., `allenai/c4`.  If changed, then `data.subdata_name` and `data.split` may have to be set as well.

These are Gaussmark specific parameters, which uses [vLLM](https://github.com/vllm-project/vllm) as a backbone for generation.  See the `sampling` parameters in `hydra_configs/hf_master.yaml` to change the sampling strategy.

