# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import omegaconf
from .base import *
from .text import *
from .wav import *

KLASS = {
    'phoneme_tokenizer': PhonemeTokenizerConditioner,
    'audio_tokenizer_wrapper': AudioTokenizerConditioner,
}

def get_condition_fuser(fuser_cfgs) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_methods = ['sum', 'cross', 'prepend', 'input_interpolate']
    fuse2cond = {k: fuser_cfgs[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfgs.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser

def get_conditioner_provider(cfg) -> ConditioningProvider:
    """Instantiate a conditioning model."""

    dict_cfg = {} if cfg is None else dict(cfg)
    conditioners: tp.Dict[str, BaseConditioner] = {}

    # import pdb; pdb.set_trace()
    for cond, cond_cfg in dict_cfg.items():
        model_args = cond_cfg.copy()
        model_type = model_args.pop('type')
        conditioners[str(cond)] = KLASS[model_type](**model_args)
    conditioner = ConditioningProvider(conditioners)
    return conditioner