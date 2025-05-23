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

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages

install_requires = [
  'accelerate',
  'codetiming',
  'datasets',
  'dill',
  'hydra-core',
  'numpy',
  'pandas',
  'peft',
  'pyarrow>=15.0.0',
  'pybind11',
  'pylatexenc',
  'ray>=2.10',
  'tensordict==0.6.0',
  'torchdata',
  'transformers',
  # 'vllm==0.7.3',
  'wandb',
  'flash-attn',
]

# TEST_REQUIRES = ['pytest', 'yapf', 'py-spy']
# PRIME_REQUIRES = ['pyext']
# GPU_REQUIRES = ['liger-kernel', 'flash-attn']

# extras_require = {
#   'test': TEST_REQUIRES,
#   'prime': PRIME_REQUIRES,
#   'gpu': GPU_REQUIRES,
# }

setup(
    name='sverl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
)