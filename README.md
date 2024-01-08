# Fast and Optimal Weight Update for Pruned Large Language Models
Official PyTorch implementation of **Fast and Optimal Weight Update for Pruned Large Language Models** as presented in (https://arxiv.org/abs/2401.02938).
This repo is copy of [Wanda](https://github.com/locuslab/wanda) repository with our additions.

--- 
## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
Below is an example command for pruning LLaMA-7B with our method, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model baffo32/decapoda-research-llama-7B-hf \
    --prune_method admm \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/admm/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`, `admm`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model baffo32/decapoda-research-llama-7B-hf \
    --prune_method admm \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/admm/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method admm \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/admm/
```

### Zero-Shot Evaluation
For evaluating zero-shot tasks, we modify the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework so that it could evaluate pruned LLM models. We provide the modified repo in [this link](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing). Make sure to download, extract and install this custom `lm_eval` package from the source code.

For reproducibility, we used [commit `df3da98`](https://github.com/EleutherAI/lm-evaluation-harness/tree/df3da98c5405deafd519c2ddca52bb7c3fe36bef) on the main branch. All tasks were evaluated on task version of 0 except for BoolQ, where the task version is 1.

On a high level, the functionality we provide is adding two arguments `pretrained_model` and `tokenizer` in this [function](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/evaluator.py#L17). We can then call this `simple_evaluate` function API from our [codebase](https://github.com/locuslab/wanda/blob/main/lib/eval.py#L148) to evaluate sparse pruned LLMs. To evaluate zero-shot tasks in addition to the WikiText perplexity, pass the `--eval_zero_shot` argument. 

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss papers/code with us through issues/emails!

boza at fmph.uniba.sk 
