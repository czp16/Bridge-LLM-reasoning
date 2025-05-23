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

from importlib.metadata import version, PackageNotFoundError
from packaging import version as vs


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)
vllm_version = None

assert vs.parse(package_version) >= vs.parse('0.6.6.post2.dev252+g8027a724')
# From 0.6.6.post2 on, vllm supports SPMD inference
# See https://github.com/vllm-project/vllm/pull/12071

from vllm import LLM
from vllm.distributed import parallel_state
from .dtensor_weight_loaders import load_dtensor_weights
