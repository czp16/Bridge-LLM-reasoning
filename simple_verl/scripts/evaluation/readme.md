# Evaluate performances on iGSM evaluation dataset
## Convert file of model weights
Since vllm loads model from files with Huggingface format (i.e., `xxx.safetensor` for model and others for tokenizer) while verl only save model with `.pt` files (e.g., the files `model_world_size_2_rank_0.pt` in `/checkpoints/{project_name}/{run_name}/global_step_100/actor/`), we need to first convert it to `xxx.safetensor` file. To achieve it, you can run
```
python simple_verl/scripts/evaluation/convert_fsdp_shard_to_safetensor.py --local_dir checkpoints/.../global_step_100/actor
```
Then it save the files to `./.../actor/huggingface`.

## Run evaluation with vllm
Then you can evaluate the model on a iGSM test set (we assume it is a `.parquet` file preprocessed by `eval_igsm.py`) by
```
export CUDA_VISIBLE_DEVICES=0
# export VLLM_USE_V1=0 # see below
python scripts/evaluation/eval_igsm --model_dir {path_you_saved_above} [other flags]
```
where `{path_you_saved_above}` looks like `./.../actor/huggingface`.

## Troubleshooting
When using vllm==0.8.1, you may encounter the error
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
``` 

You can set `export VLLM_USE_V1=0` to disable engine V1.