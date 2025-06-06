# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""
import warnings

__all__ = ['hf_tokenizer']

llama_chat_template = """
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        kwargs['eos_token'] = '<end_of_turn>'
        kwargs['eos_token_id'] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    
    if isinstance(name_or_path, str) and 'llama' in name_or_path:
        print("Reset llama chat template.")
        tokenizer.chat_template = llama_chat_template
    elif tokenizer.chat_template is None:
        raise ValueError(f"Tokenizer {name_or_path} does not have chat template.")
        
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer