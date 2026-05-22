# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2020- The Blackjax Authors.
# Ported to tengri from github.com/handley-lab/blackjax (nested_sampling branch).
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
"""Type aliases and base types for the nested sampling package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray
ArrayTree = Any
ArrayLikeTree = Any
PRNGKey = jnp.ndarray


class SamplingAlgorithm(NamedTuple):
    """A pair of (init, step) functions defining a sampling algorithm."""

    init: Callable
    step: Callable
