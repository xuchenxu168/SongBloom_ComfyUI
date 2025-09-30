import torch
import os,sys
from transformers.utils import is_flash_attn_2_available
from transformers.models.llama import LlamaModel, LlamaConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartConfig
import warnings
# from transformers.models.musicgen.modeling_musicgen import MusicgenModel, MusicgenDecoder, MusicgenDecoderConfig # 用的就是BartDecoder，但是没有cross-attn

try:
    assert is_flash_attn_2_available()
    assert torch.cuda.get_device_capability(torch.device("cuda")) >= (8, 0)
    assert os.environ.get("DISABLE_FLASH_ATTN",'0') != "1"
    _enable_flash_attention = True
except:
    _enable_flash_attention = False

if not _enable_flash_attention:
    warnings.warn("Not support flash-attn!")

def get_backend(name, dim, num_heads, num_layers, hidden_scale, init_std=0.02, rope_theta=10000,):
    # SA (causal) - FF
    if name == 'llama':
        model_cfg = LlamaConfig(
            hidden_size=dim,
            intermediate_size=dim * hidden_scale,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=num_heads,
            vocab_size=dim,
            use_cache=False,
            max_position_embeddings=4096,
            hidden_act="silu",
            initializer_range=init_std,
            rope_theta=rope_theta,
            _attn_implementation="flash_attention_2" if _enable_flash_attention else "eager",
        )
        model = LlamaModel(model_cfg)
        
    # SA -FF
    elif name == 'bart_enc':
        model_cfg = BartConfig(
            d_model=dim,
            max_position_embeddings=4096,
            dropout=0., 
            use_cache=False,
            _attn_implementation="flash_attention_2" if _enable_flash_attention else "eager",
            activation_function='gelu',
            # for BartEncoder
            encoder_layers=num_layers, 
            encoder_attention_heads=num_heads,
            init_std=init_std,
            encoder_ffn_dim=dim * hidden_scale,
        )
        model = BartEncoder(model_cfg)
        
    # SA - CA - FF
    elif name == 'bart_dec':
        model_cfg = BartConfig(
            d_model=dim,
            max_position_embeddings=4096,
            dropout=0., 
            use_cache=False,
            _attn_implementation="flash_attention_2" if _enable_flash_attention else "eager",
            activation_function='gelu',
            # for BartDecoder
            decoder_layers=num_layers,
            decoder_attention_heads=num_heads,
            decoder_ffn_dim=dim * hidden_scale,
        )
        model = BartDecoder(model_cfg)

    else:
        raise NotImplementedError

    delattr(model, "embed_tokens")
    return model