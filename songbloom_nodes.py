"""
SongBloom ComfyUI Nodes

Main implementation of SongBloom nodes for ComfyUI.
"""

import os
import sys
import torch
import torchaudio
import json
import tempfile
import traceback
import logging
import warnings
import numpy as np
import random
import uuid
from typing import Dict, List, Tuple, Optional, Any
import folder_paths
from openai import OpenAI
import os
import json
import requests

# Set up logging for SongBloom plugin
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SongBloom")

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Add SongBloom to path
songbloom_path = os.path.join(os.path.dirname(__file__), "SongBloom")
if songbloom_path not in sys.path:
    sys.path.insert(0, songbloom_path)

try:
    from omegaconf import OmegaConf, DictConfig
    from huggingface_hub import hf_hub_download
    
    # Import SongBloom components
    os.environ['DISABLE_FLASH_ATTN'] = "1"
    from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
    from SongBloom.g2p.lyric_common import key2processor, symbols, LABELS
    
    SONGBLOOM_AVAILABLE = True
except ImportError as e:
    print(f"SongBloom dependencies not available: {e}")
    SONGBLOOM_AVAILABLE = False

# Model repository mapping
NAME2REPO = {
    "songbloom_full_150s": "CypressYang/SongBloom",
    "songbloom_full_150s_dpo": "CypressYang/SongBloom"
}

# Error handling utilities
class SongBloomError(Exception):
    """Base exception for SongBloom plugin errors"""
    pass

class ModelLoadError(SongBloomError):
    """Error loading SongBloom model"""
    pass

class AudioProcessingError(SongBloomError):
    """Error processing audio"""
    pass

class LyricProcessingError(SongBloomError):
    """Error processing lyrics"""
    pass

class GenerationError(SongBloomError):
    """Error during song generation"""
    pass

def safe_execute(func, error_type=SongBloomError, default_return=None):
    """Safely execute a function with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            if default_return is not None:
                return default_return
            raise error_type(f"Error in {func.__name__}: {str(e)}")
    return wrapper

def validate_audio_input(audio, required_duration=None):
    """Validate audio input format and properties"""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise AudioProcessingError("Invalid audio format. Expected dict with 'waveform' and 'sample_rate'")

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if not isinstance(waveform, torch.Tensor):
        raise AudioProcessingError("Waveform must be a torch.Tensor")

    if waveform.numel() == 0:
        raise AudioProcessingError("Empty audio waveform")

    if required_duration:
        actual_duration = waveform.shape[-1] / sample_rate
        if actual_duration < required_duration * 0.5:  # Allow 50% tolerance
            raise AudioProcessingError(f"Audio too short: {actual_duration:.2f}s, required: {required_duration}s")

    return True

def validate_lyrics_input(lyrics):
    """Validate lyrics input"""
    if not isinstance(lyrics, str):
        raise LyricProcessingError("Lyrics must be a string")

    if not lyrics.strip():
        raise LyricProcessingError("Lyrics cannot be empty")

    # Check for minimum structure
    if not any(tag in lyrics for tag in ['[verse]', '[chorus]', '[intro]', '[outro]']):
        logger.warning("No structure tags found in lyrics. This may affect generation quality.")

    return True

def get_local_models():
    """获取本地可用的SongBloom模型"""
    models_dir = os.path.join(folder_paths.models_dir, "SongBloom")
    local_models = []

    if os.path.exists(models_dir):
        # 查找.pt文件
        for file in os.listdir(models_dir):
            if file.endswith('.pt') and 'songbloom' in file.lower():
                model_name = file.replace('.pt', '')
                local_models.append(model_name)

    # 如果没有找到本地模型，返回默认选项
    if not local_models:
        local_models = ["songbloom_full_150s", "songbloom_full_150s_dpo"]

    return sorted(local_models)

def get_audio_files():
    """获取可用的音频文件列表 - 搜索更广泛的目录"""
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac', '.wma']
    audio_files = ["无 (不使用音频提示)"]

    # 搜索多个可能的音频目录
    search_dirs = []

    # 1. ComfyUI相关目录
    try:
        if hasattr(folder_paths, 'get_input_directory'):
            input_dir = folder_paths.get_input_directory()
            if input_dir and os.path.exists(input_dir):
                search_dirs.append(("ComfyUI输入", input_dir))
    except:
        pass

    try:
        models_audio_dir = os.path.join(folder_paths.models_dir, "audio")
        if os.path.exists(models_audio_dir):
            search_dirs.append(("ComfyUI音频", models_audio_dir))
    except:
        pass

    # 2. 系统常见音频目录
    common_audio_dirs = [
        ("用户音乐", os.path.expanduser("~/Music")),
        ("用户音乐", os.path.expanduser("~/音乐")),
        ("文档音乐", os.path.expanduser("~/Documents/Music")),
        ("桌面", os.path.expanduser("~/Desktop")),
        ("下载", os.path.expanduser("~/Downloads")),
        ("下载", os.path.expanduser("~/下载")),
    ]

    for desc, music_dir in common_audio_dirs:
        if os.path.exists(music_dir):
            search_dirs.append((desc, music_dir))

    # 3. Windows系统音频目录
    if os.name == 'nt':  # Windows
        windows_audio_dirs = [
            ("公共音乐", "C:/Users/Public/Music"),
            ("系统音乐", "C:/Windows/Media"),
        ]
        for desc, dir_path in windows_audio_dirs:
            if os.path.exists(dir_path):
                search_dirs.append((desc, dir_path))

    # 4. 添加一些常见的音频软件目录
    audio_software_dirs = [
        ("FL Studio", os.path.expanduser("~/Documents/Image-Line/FL Studio")),
        ("Audacity", os.path.expanduser("~/Documents/Audacity")),
        ("录音", os.path.expanduser("~/Documents/录音")),
    ]

    for desc, dir_path in audio_software_dirs:
        if os.path.exists(dir_path):
            search_dirs.append((desc, dir_path))

    # 5. 当前工作目录和ComfyUI根目录
    current_dir = os.getcwd()
    search_dirs.append(("当前目录", current_dir))

    # 搜索音频文件
    found_files = set()  # 使用set避免重复

    for desc, search_dir in search_dirs:
        try:
            logger.debug(f"Searching audio files in {desc}: {search_dir}")
            for root, dirs, files in os.walk(search_dir):
                # 限制搜索深度，避免搜索太深
                level = root.replace(search_dir, '').count(os.sep)
                if level >= 3:  # 最多搜索3层深度
                    dirs[:] = []  # 不再深入搜索
                    continue

                # 跳过一些不必要的目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and
                          d.lower() not in ['__pycache__', 'node_modules', 'cache', 'temp', 'tmp']]

                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        full_path = os.path.abspath(os.path.join(root, file))

                        # 避免重复添加
                        if full_path not in found_files:
                            found_files.add(full_path)
                            audio_files.append(full_path)

                        # 限制文件数量，避免列表过长
                        if len(audio_files) >= 100:
                            break

                if len(audio_files) >= 100:
                    break
        except Exception as e:
            logger.debug(f"Error searching directory {desc} ({search_dir}): {e}")
            continue

    # 按文件名排序，方便查找
    if len(audio_files) > 1:
        # 保持"无"选项在第一位，其他按文件名排序
        no_audio_option = audio_files[0]
        other_files = sorted(audio_files[1:], key=lambda x: os.path.basename(x).lower())
        audio_files = [no_audio_option] + other_files

    logger.info(f"Found {len(audio_files)-1} audio files in {len(search_dirs)} directories")
    return audio_files

class SongBloomModelLoader:
    """
    Node for loading SongBloom models (local or from HuggingFace Hub)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 动态获取可用模型
        available_models = get_local_models()

        return {
            "required": {
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "songbloom_full_150s"
                }),
                "dtype": (["float32", "bfloat16"], {
                    "default": "float32"
                }),
                "lyric_processor": (["pinyin", "phoneme", "none"], {
                    "default": "phoneme"
                }),
                "load_mode": (["local_first", "local_only", "download_only"], {
                    "default": "local_first"
                }),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("SONGBLOOM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SongBloom/Models"
    
    def load_model(self, model_name: str, dtype: str, lyric_processor: str = "phoneme", load_mode: str = "local_first", force_reload: bool = False):
        """Load SongBloom model with local and remote support"""
        if not SONGBLOOM_AVAILABLE:
            raise ModelLoadError("SongBloom dependencies not available. Please install requirements with: pip install -r requirements.txt")

        try:
            logger.info(f"Loading SongBloom model: {model_name} with dtype: {dtype}, mode: {load_mode}")
            logger.info(f"Lyric processor: {lyric_processor}")

            # Validate inputs
            if dtype not in ['float32', 'bfloat16']:
                raise ModelLoadError(f"Invalid dtype: {dtype}. Must be 'float32' or 'bfloat16'")

            # Check GPU availability for bfloat16
            if dtype == 'bfloat16' and not torch.cuda.is_available():
                logger.warning("bfloat16 requested but CUDA not available, falling back to float32")
                dtype = 'float32'

            # Set up models directory
            models_dir = os.path.join(folder_paths.models_dir, "SongBloom")
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Using models directory: {models_dir}")

            # Try to load model based on mode
            if load_mode == "local_only":
                success = self._load_local_model(model_name, models_dir)
                if not success:
                    raise ModelLoadError(f"Local model not found: {model_name}")
            elif load_mode == "download_only":
                self._download_model_files(model_name, models_dir, force_reload)
            else:  # local_first
                success = self._load_local_model(model_name, models_dir)
                if not success:
                    logger.info("Local model not found, downloading from HuggingFace...")
                    self._download_model_files(model_name, models_dir, force_reload)

            # Load configuration
            cfg_path = os.path.join(models_dir, f"{model_name}.yaml")
            if not os.path.exists(cfg_path):
                # Try to find any yaml file if exact match not found
                yaml_files = [f for f in os.listdir(models_dir) if f.endswith('.yaml')]
                if yaml_files:
                    cfg_path = os.path.join(models_dir, yaml_files[0])
                    logger.info(f"Using configuration file: {yaml_files[0]}")
                else:
                    raise ModelLoadError(f"No configuration file found in {models_dir}")

            cfg = self._load_config(cfg_path, models_dir)

            # Update model path in config
            model_path = os.path.join(models_dir, f"{model_name}.pt")
            if not os.path.exists(model_path):
                raise ModelLoadError(f"Model file not found: {model_path}")

            cfg.pretrained_path = model_path

            # Ensure lyric processor is set based on UI selection
            try:
                from omegaconf import OmegaConf as _OC
            except Exception:
                _OC = None

            if not hasattr(cfg, 'train_dataset'):
                if _OC is not None:
                    cfg.train_dataset = _OC.create({})
                else:
                    cfg.train_dataset = type('obj', (), {})()

            # Map 'none' to None for passthrough behavior
            lp_value = None if lyric_processor == 'none' else lyric_processor
            try:
                # Validate value against available processors if possible
                if lp_value is not None and lp_value not in key2processor.keys():
                    logger.warning(f"Unknown lyric_processor '{lp_value}', falling back to 'phoneme'")
                    lp_value = 'phoneme'
            except Exception:
                pass
            cfg.train_dataset.lyric_processor = lp_value

            # Validate configuration
            if not hasattr(cfg, 'max_dur'):
                logger.warning("Configuration missing max_dur, using default: 150")
                cfg.max_dur = 150
            if not hasattr(cfg, 'inference'):
                logger.warning("Configuration missing inference settings, using high-quality defaults")
                cfg.inference = {
                    'cfg_coef': 3.0,      # 提高CFG系数以获得更好的质量
                    'steps': 100,         # 增加步数以获得更精细的生成
                    'top_k': 100,         # 降低top_k以减少随机性
                    'use_sampling': True,
                    'dit_cfg_type': 'h'
                }

            # Validate VAE configuration
            if not hasattr(cfg, 'vae'):
                logger.warning("Configuration missing VAE settings, using defaults")
                # StableVAE需要vae_ckpt和vae_cfg参数
                vae_ckpt_path = os.path.join(models_dir, 'vae.pt')
                vae_cfg_path = os.path.join(models_dir, 'vae_config.json')

                # 如果文件不存在，尝试查找
                if not os.path.exists(vae_ckpt_path):
                    # 查找VAE相关文件，支持多种扩展名
                    vae_extensions = ['.pt', '.ckpt', '.pth']
                    vae_files = []

                    for ext in vae_extensions:
                        files = [f for f in os.listdir(models_dir)
                                if ('vae' in f.lower() or 'autoencoder' in f.lower()) and f.endswith(ext)]
                        vae_files.extend(files)

                    if vae_files:
                        vae_ckpt_path = os.path.join(models_dir, vae_files[0])
                        logger.info(f"Found VAE checkpoint: {vae_files[0]}")
                    else:
                        logger.error(f"VAE checkpoint not found in {models_dir}")
                        raise ModelLoadError(f"VAE checkpoint file not found in {models_dir}")

                if not os.path.exists(vae_cfg_path):
                    # 查找VAE配置文件
                    cfg_files = [f for f in os.listdir(models_dir) if 'vae' in f.lower() and f.endswith('.json')]
                    if cfg_files:
                        vae_cfg_path = os.path.join(models_dir, cfg_files[0])
                        logger.info(f"Found VAE config: {cfg_files[0]}")
                    else:
                        logger.error(f"VAE config not found in {models_dir}")
                        raise ModelLoadError(f"VAE config file not found in {models_dir}")

                cfg.vae = {
                    'vae_ckpt': vae_ckpt_path,
                    'vae_cfg': vae_cfg_path,
                    'sr': 48000
                }

            # Validate other required configurations
            if not hasattr(cfg, 'model'):
                logger.warning("Configuration missing model settings, using defaults")
                cfg.model = {
                    'name': 'songbloom_model',
                    'sample_rate': 48000,
                    'max_seq_len': 8192,
                    'embed_dim': 1024,
                    'num_heads': 16,
                    'num_layers': 24,
                    'dropout': 0.1
                }

            if not hasattr(cfg, 'tokenizer'):
                logger.warning("Configuration missing tokenizer settings, using defaults")
                cfg.tokenizer = {
                    'name': 'songbloom_tokenizer',
                    'vocab_size': 50000,
                    'max_length': 512,
                    'pad_token': '<pad>',
                    'unk_token': '<unk>',
                    'bos_token': '<bos>',
                    'eos_token': '<eos>'
                }

            # Adjust max duration
            cfg.max_dur = cfg.max_dur + 20

            # Set dtype
            torch_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
            logger.info(f"🔧 Using dtype: {torch_dtype} for model building")

            # Build model
            logger.info("Building SongBloom model...")
            model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=torch_dtype)
            model.set_generation_params(**cfg.inference)
            
            # Verify model components dtype
            if hasattr(model, 'compression_model') and model.compression_model is not None:
                comp_dtype = next(model.compression_model.parameters()).dtype
                logger.info(f"🔧 Compression model dtype: {comp_dtype}")
                # Force convert to target dtype if different
                if comp_dtype != torch_dtype:
                    logger.info(f"🔄 Converting compression model from {comp_dtype} to {torch_dtype}")
                    model.compression_model = model.compression_model.to(dtype=torch_dtype)
            if hasattr(model, 'diffusion') and model.diffusion is not None:
                diff_dtype = next(model.diffusion.parameters()).dtype
                logger.info(f"🔧 Diffusion model dtype: {diff_dtype}")
                # Force convert to target dtype if different
                if diff_dtype != torch_dtype:
                    logger.info(f"🔄 Converting diffusion model from {diff_dtype} to {torch_dtype}")
                    model.diffusion = model.diffusion.to(dtype=torch_dtype)

            logger.info(f"Successfully loaded SongBloom model: {model_name}")
            return (model,)

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading SongBloom model: {e}")
            logger.debug(traceback.format_exc())
            raise ModelLoadError(f"Failed to load SongBloom model: {str(e)}")
    
    def _load_local_model(self, model_name: str, models_dir: str):
        """检查并验证本地模型文件"""
        try:
            # 检查主模型文件是否存在
            model_file = os.path.join(models_dir, f"{model_name}.pt")
            if not os.path.exists(model_file):
                logger.info(f"Local model file not found: {model_file}")
                return False

            logger.info(f"Found local model file: {model_file}")

            # 可选文件（如果不存在会尝试下载）
            optional_files = [
                f"{model_name}.yaml",
                "stable_audio_1920_vae.json",
                "autoencoder_music_dsp1920.ckpt",
                "vocab_g2p.yaml"
            ]

            # 检查其他必需文件，如果不存在则尝试下载
            missing_files = []
            for filename in optional_files:
                file_path = os.path.join(models_dir, filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)

            if missing_files:
                logger.info(f"Missing optional files: {missing_files}, will try to download")
                try:
                    self._download_optional_files(missing_files, models_dir)
                except Exception as e:
                    logger.warning(f"Failed to download optional files: {e}")
                    # 继续使用本地模型，即使某些文件缺失

            return True

        except Exception as e:
            logger.error(f"Error checking local model: {e}")
            return False

    def _download_optional_files(self, filenames: list, models_dir: str):
        """下载缺失的可选文件"""
        # 使用默认仓库下载缺失文件
        repo_id = "CypressYang/SongBloom"

        for filename in filenames:
            try:
                logger.info(f"Downloading optional file: {filename}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Successfully downloaded: {filename}")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")

    def _download_model_files(self, model_name: str, cache_dir: str, force_reload: bool):
        """Download model files from HuggingFace Hub with error handling"""
        # 检查模型名称是否在已知仓库中
        if model_name in NAME2REPO:
            repo_id = NAME2REPO[model_name]
        else:
            # 对于本地模型，使用默认仓库
            repo_id = "CypressYang/SongBloom"
            logger.info(f"Using default repository for model: {model_name}")

        files_to_download = [
            f"{model_name}.yaml",
            f"{model_name}.pt",
            "stable_audio_1920_vae.json",
            "autoencoder_music_dsp1920.ckpt",
            "vocab_g2p.yaml"
        ]

        for filename in files_to_download:
            local_path = os.path.join(cache_dir, filename)
            if force_reload or not os.path.exists(local_path):
                try:
                    logger.info(f"Downloading {filename} from {repo_id}...")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    # 对于模型文件，如果下载失败则抛出错误
                    if filename.endswith('.pt'):
                        raise ModelLoadError(f"Failed to download model file {filename}: {str(e)}")
                    else:
                        logger.warning(f"Failed to download optional file {filename}: {str(e)}")
            else:
                logger.debug(f"Using cached file: {filename}")

        # 验证主模型文件存在
        model_file = os.path.join(cache_dir, f"{model_name}.pt")
        if not os.path.exists(model_file):
            raise ModelLoadError(f"Model file missing after download: {model_name}.pt")
    
    def _load_config(self, cfg_file: str, parent_dir: str) -> DictConfig:
        """Load configuration file with safe resolver registration"""
        # 安全注册OmegaConf resolvers，避免重复注册错误
        resolvers = {
            "eval": lambda x: eval(x),
            "concat": lambda *x: [xxx for xx in x for xxx in xx],
            "get_fname": lambda x: os.path.splitext(os.path.basename(x))[0],
            "load_yaml": lambda x: OmegaConf.load(x),
            "dynamic_path": lambda x: x.replace("???", parent_dir)
        }

        for name, resolver in resolvers.items():
            try:
                OmegaConf.register_new_resolver(name, resolver)
                logger.debug(f"Registered OmegaConf resolver: {name}")
            except ValueError as e:
                if "already registered" in str(e):
                    logger.debug(f"OmegaConf resolver '{name}' already registered, skipping")
                else:
                    logger.warning(f"Failed to register resolver '{name}': {e}")

        return OmegaConf.load(open(cfg_file, 'r'))


class SongBloomLyricProcessor:
    """
    Node for processing and validating lyrics for SongBloom
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] City lights flicker through the car window. Dreams pass fast where the lost ones go. Neon signs echo stories untold. I chase shadows while the night grows cold , [chorus] Run with me down the empty street. Where silence and heartbeat always meet. Every breath. a whispered vow. We are forever. here and now"
                }),
                "validate_format": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_lyrics", "validation_info")
    FUNCTION = "process_lyrics"
    CATEGORY = "SongBloom/Text"
    
    def process_lyrics(self, lyrics: str, validate_format: bool = True):
        """Process and validate lyrics with comprehensive error handling"""
        try:
            # Input validation
            validate_lyrics_input(lyrics)

            # Basic cleanup
            processed_lyrics = lyrics.strip()

            # Remove excessive whitespace
            processed_lyrics = ' '.join(processed_lyrics.split())

            # Ensure proper spacing around structure tags
            import re
            processed_lyrics = re.sub(r'\s*(\[[^\]]+\])\s*', r' \1 ', processed_lyrics)
            processed_lyrics = re.sub(r'\s+', ' ', processed_lyrics).strip()

            validation_info = "Lyrics processed successfully."

            if validate_format:
                validation_info = self._validate_lyric_format(processed_lyrics)

            logger.info(f"Processed lyrics: {len(processed_lyrics)} characters")
            return (processed_lyrics, validation_info)

        except LyricProcessingError as e:
            logger.error(f"Lyric processing error: {e}")
            return (lyrics, f"Lyric processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing lyrics: {e}")
            logger.debug(traceback.format_exc())
            return (lyrics, f"Unexpected error processing lyrics: {str(e)}")
    
    def _validate_lyric_format(self, lyrics: str) -> str:
        """Validate lyric format according to SongBloom requirements"""
        issues = []
        
        # Check for structure tags
        structure_tags = ['[intro]', '[verse]', '[chorus]', '[bridge]', '[inst]', '[outro]']
        found_tags = []
        for tag in structure_tags:
            if tag in lyrics:
                found_tags.append(tag)
        
        if not found_tags:
            issues.append("No structure tags found. Please include tags like [verse], [chorus], etc.")
        
        # Check for proper separators
        if ',' not in lyrics and len(found_tags) > 1:
            issues.append("Missing section separators (,). Use ',' to separate sections.")
        
        # Check for sentence separators
        vocal_sections = [section.strip() for section in lyrics.split(',')]
        for section in vocal_sections:
            if any(tag in section for tag in ['[verse]', '[chorus]', '[bridge]']):
                if '.' not in section:
                    issues.append(f"Missing sentence separators (.) in vocal section: {section[:50]}...")
        
        if issues:
            return "Validation issues found:\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            return "Lyrics format is valid."


class SongBloomLyricGenerator:
    """
    Node for generating structured lyrics from text descriptions
    Automatically formats text into SongBloom-compatible lyric structure
    """

    def __init__(self):
        self.lyric_tags = ["[intro]", "[verse]", "[chorus]", "[bridge]", "[outro]", "[inst]"]
        self.max_lyric_length = 2000  # Maximum characters for lyrics

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "写一首关于青春回忆的歌曲，包含对过去美好时光的怀念和对未来的憧憬"
                }),
                "style": (["pop", "rock", "ballad", "folk", "electronic", "jazz", "classical", "hip-hop", "country", "blues", "r&b", "reggae"], {
                    "default": "pop"
                }),
                "mood": (["happy", "sad", "nostalgic", "energetic", "peaceful", "romantic", "melancholic", "angry", "hopeful", "mysterious", "dramatic", "playful"], {
                    "default": "nostalgic"
                }),
                "language": (["chinese", "english", "mixed", "japanese", "korean", "spanish", "french"], {
                    "default": "chinese"
                }),
                "structure": (["simple", "standard", "complex", "custom"], {
                    "default": "standard"
                }),
                "length": (["short", "medium", "long", "very_long"], {
                    "default": "medium"
                }),
                "format_style": (["traditional", "modern", "minimal", "detailed"], {
                    "default": "traditional",
                    "tooltip": "歌词格式风格：traditional/modern/minimal/detailed"
                }),
            },
            "optional": {
                "use_deepseek": ("BOOLEAN", {"default": True}),
                "deepseek_api_key": ("STRING", {"default": "", "multiline": False}),
                "deepseek_model": (["DeepSeek-V3"], {"default": "DeepSeek-V3"}),
                "deepseek_system_prompt": ("STRING", {"default": "## 角色设定 (System Role):\n你是一名专业的歌词专家。你的任务是根据提示词，谱写一首非常专业的精湛的歌词，歌词要丰富多彩，情真意切。\n## 任务说明 (User Instruction):\n请仔细观察提供的歌词提示内容，并谱写一段详细、具体、富有创造性的符合歌词格式的高质量歌词，\n## 输出要求 (Output Requirements):\n**   **语言**： 严格使用中文。\n**   **格式**： 严格按照下面示例格式谱写歌词，严格按照格式，不要分段。\n## 只生成歌词，无任何其它的词\n\n歌词严格按照下面的格式：\n[intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] 高质量音频生成测试. 使用优化参数设置. CFG系数提高到三点零. 步数增加到一百步 , [chorus] 清晰的音质表现. 丰富的音乐细节. 每个音符都精准. 高保真音频体验 , [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] , [verse] 降低随机性参数. 提高生成稳定性. 更长的推理时间. 换来更好的质量 , [bridge] 当音乐响起的时候. 每一个细节都清晰. 从低音到高音频段. 都有完美的表现 , [chorus] 清晰的音质表现. 丰富的音乐细节. 每个音符都精准. 高保真音频体验 , [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro]", "multiline": True}),
                "custom_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "自定义标签，用逗号分隔，如：intro,verse,chorus,bridge,outro"
                }),
                "theme_keywords": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "主题关键词，用逗号分隔"
                }),
                "user_requirements": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "用户特殊要求，如：每行字数限制/押韵要求/特定词汇等"
                }),
                "rhyme_scheme": (["none", "aabb", "abab", "abcb", "free"], {
                    "default": "none",
                    "tooltip": "押韵模式"
                }),
                "line_length": (["short", "medium", "long", "mixed"], {
                    "default": "medium",
                    "tooltip": "每行长度"
                }),
                "repetition_style": (["none", "minimal", "moderate", "heavy"], {
                    "default": "moderate",
                    "tooltip": "重复风格"
                }),
                "emotional_intensity": (["low", "medium", "high", "extreme"], {
                    "default": "medium",
                    "tooltip": "情感强度"
                }),
                "target_audience": (["children", "teenagers", "adults", "elderly", "general"], {
                    "default": "general",
                    "tooltip": "目标受众"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_lyrics",)
    FUNCTION = "generate_lyrics"
    CATEGORY = "SongBloom/Lyrics"

    def generate_lyrics(self, text_prompt: str, style: str = "pop", mood: str = "nostalgic",
                        language: str = "chinese", structure: str = "standard",
                        length: str = "medium", format_style: str = "traditional",
                        custom_tags: str = "", theme_keywords: str = "",
                        user_requirements: str = "", rhyme_scheme: str = "none",
                        line_length: str = "medium", repetition_style: str = "moderate",
                        emotional_intensity: str = "medium", target_audience: str = "general",
                        use_deepseek: bool = True, deepseek_api_key: str = "", deepseek_model: str = "DeepSeek-V3",
                        deepseek_system_prompt: str = ""):
        """Generate structured lyrics from text description with enhanced user control"""
        try:
            if use_deepseek:
                try:
                    sys_prompt = deepseek_system_prompt.strip() or (
                        "你是一名专业的歌词专家。请把用户提供的主题与要求，严格转换为 SongBloom 所需的结构化歌词格式。" \
                        "输出仅包含歌词文本；使用中文；段落标签必须是 [intro]/[verse]/[chorus]/[bridge]/[inst]/[outro]；段落之间用英文逗号分隔，句子用英文句号分隔。"
                    )

                    meta = []
                    if style: meta.append(f"style={style}")
                    if mood: meta.append(f"mood={mood}")
                    if language: meta.append(f"language={language}")
                    if theme_keywords: meta.append(f"keywords={theme_keywords}")
                    if user_requirements: meta.append(f"requirements={user_requirements}")
                    if custom_tags: meta.append(f"custom_tags={custom_tags}")
                    if rhyme_scheme != "none": meta.append(f"rhyme={rhyme_scheme}")
                    ds_user_prompt = text_prompt + ("\n\n# meta: " + ", ".join(meta) if meta else "")

                    # 读取 API Key：节点输入 > 环境变量 > DeepseekAP-config.json > config.yaml
                    api_key = (deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
                    if not api_key:
                        try:
                            json_path = os.path.join(os.path.dirname(__file__), "DeepseekAP-config.json")
                            if os.path.exists(json_path):
                                with open(json_path, 'r', encoding='utf-8') as jf:
                                    j = json.load(jf) or {}
                                api_key = (j.get('api_key') or "").strip()
                        except Exception:
                            pass
                    if not api_key:
                        try:
                            import yaml
                            cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                            with open(cfg_path, 'r', encoding='utf-8') as f:
                                data = yaml.safe_load(f) or {}
                            api_key = ((data.get('deepseek') or {}).get('api_key') or "").strip()
                        except Exception:
                            pass

                    if api_key:
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                        completion = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": ds_user_prompt},
                            ],
                            stream=False,
                        )
                        ds_text = completion.choices[0].message.content if completion and completion.choices else ""
                        if ds_text and isinstance(ds_text, str) and len(ds_text.strip()) > 0:
                            formatted_lyrics = self._format_lyrics(ds_text.strip(), format_style, rhyme_scheme, target_audience, language)
                            logger.info(f"✅ Generated lyrics by DeepSeek: {len(formatted_lyrics)} characters")
                            return (formatted_lyrics,)
                except Exception as e:
                    logger.warning(f"DeepSeek 生成失败，回退本地规则: {e}")
            logger.info(f"🎵 Generating lyrics: style={style}, mood={mood}, language={language}")

            generated_lyrics = self._create_lyric_structure(
                text_prompt, style, mood, language, structure, length, format_style,
                custom_tags, theme_keywords, user_requirements, rhyme_scheme,
                line_length, repetition_style, emotional_intensity, target_audience
            )

            formatted_lyrics = self._format_lyrics(
                generated_lyrics, format_style, rhyme_scheme, target_audience, language
            )

            logger.info(f"✅ Generated lyrics: {len(formatted_lyrics)} characters")
            return (formatted_lyrics,)

        except Exception as e:
            logger.error(f"Error generating lyrics: {e}")
            fallback_lyrics = self._get_fallback_lyrics(style, mood, language)
            return (fallback_lyrics,)

    def _create_lyric_structure(self, text_prompt: str, style: str, mood: str,
                                language: str, structure: str, length: str, format_style: str,
                                custom_tags: str, theme_keywords: str, user_requirements: str,
                                rhyme_scheme: str, line_length: str, repetition_style: str,
                                emotional_intensity: str, target_audience: str):
        """Create lyric structure based on enhanced parameters"""

        structures = {
            "simple": ["[intro]", "[verse]", "[chorus]", "[outro]"],
            "standard": ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
            "complex": ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"]
        }

        lengths = {
            "short": {"lines_per_section": 2, "words_per_line": 4},
            "medium": {"lines_per_section": 4, "words_per_line": 6},
            "long": {"lines_per_section": 6, "words_per_line": 8}
        }

        structure_tags = structures.get(structure, structures["standard"])
        length_config = lengths.get(length, lengths["medium"])

        if custom_tags.strip():
            custom_tag_list = [tag.strip() for tag in custom_tags.split(",")]
            structure_tags = [f"[{tag}]" if not tag.startswith("[") else tag for tag in custom_tag_list]

        lyrics = []
        for tag in structure_tags:
            section_lyrics = self._generate_section_lyrics(
                tag, text_prompt, style, mood, language, length_config, theme_keywords
            )
            lyrics.append(tag)
            lyrics.extend(section_lyrics)

        if not any('[outro]' in line for line in lyrics):
            outro_lyrics = self._generate_section_lyrics(
                "[outro]", text_prompt, style, mood, language, length_config, theme_keywords
            )
            lyrics.extend(outro_lyrics)

        lyrics.append("[outro]")
        return "\n".join(lyrics)

    def _generate_section_lyrics(self, tag: str, text_prompt: str, style: str,
                                 mood: str, language: str, length_config: dict, theme_keywords: str):
        """Generate lyrics for a specific section"""

        section_templates = {
            "[intro]": {
                "chinese": ["轻柔的旋律响起", "回忆如潮水般涌来", "时光斑驳月如歌", "心中的故事开始"],
                "english": ["Soft melody begins", "Memories come flooding back", "Time flows like a gentle stream", "The story in my heart starts"],
                "mixed": ["轻柔的旋律 softly plays", "回忆 memories flow", "时光 time tells stories", "心中的 heart's melody"]
            },
            "[verse]": {
                "chinese": ["那些年我们一起走过", "青春岁月如诗如画", "梦想在心中发芽", "未来在远方等待"],
                "english": ["Those years we walked together", "Youthful days like poetry", "Dreams sprouting in our hearts", "Future waiting in the distance"],
                "mixed": ["那些年 those years together", "青春 youth like poetry", "梦想 dreams in hearts", "未来 future awaits"]
            },
            "[chorus]": {
                "chinese": ["这是我们的青春之歌", "唱出心中的梦想", "无论走到哪里", "永远记得这一刻"],
                "english": ["This is our song of youth", "Singing dreams from our hearts", "No matter where we go", "We'll always remember this moment"],
                "mixed": ["这是我们的 this is our song", "唱出梦想 singing dreams", "无论走到 wherever we go", "永远记得 always remember"]
            },
            "[bridge]": {
                "chinese": ["当夜幕降临", "星光点点", "我们依然相信", "美好的明天"],
                "english": ["When night falls", "Stars twinkle", "We still believe", "In beautiful tomorrow"],
                "mixed": ["当夜幕降临 when night falls", "星光点点 stars twinkle", "我们相信 we believe", "美好明天 beautiful tomorrow"]
            },
            "[outro]": {
                "chinese": ["歌声渐渐远去", "回忆永远珍藏", "这是我们的故事", "青春永不散场"],
                "english": ["The song fades away", "Memories forever treasured", "This is our story", "Youth never ends"],
                "mixed": ["歌声渐渐 the song fades", "回忆珍藏 memories treasured", "这是我们的 this is our story", "青春永不 ends"]
            },
            "[inst]": {
                "chinese": ["[乐器独奏]", "[音乐间奏]", "[旋律延续]", "[节奏变化]"],
                "english": ["[Instrumental solo]", "[Musical interlude]", "[Melody continues]", "[Rhythm changes]"],
                "mixed": ["[乐器独奏 instrumental]", "[音乐间奏 musical interlude]", "[旋律延续 melody continues]", "[节奏变化 rhythm changes]"]
            }
        }

        base_templates = section_templates.get(tag, section_templates["[verse]"])
        templates = base_templates.get(language, base_templates["chinese"])

        lines = []
        lines_per_section = length_config["lines_per_section"]
        for i in range(lines_per_section):
            if i < len(templates):
                line = templates[i]
            else:
                line = self._generate_custom_line(tag, mood, style, language, theme_keywords)
            lines.append(line)
        return lines

    def _generate_custom_line(self, tag: str, mood: str, style: str, language: str, theme_keywords: str):
        """Generate a custom line based on mood, style, and theme"""

        mood_words = {
            "happy": {"chinese": ["快乐", "欢笑", "阳光", "美好"], "english": ["happy", "joy", "sunshine", "beautiful"]},
            "sad": {"chinese": ["忧伤", "泪水", "离别", "思念"], "english": ["sad", "tears", "farewell", "missing"]},
            "nostalgic": {"chinese": ["回忆", "过去", "时光", "怀念"], "english": ["memories", "past", "time", "nostalgia"]},
            "energetic": {"chinese": ["激情", "活力", "奔跑", "自由"], "english": ["passion", "energy", "running", "freedom"]},
            "peaceful": {"chinese": ["安静", "平和", "温柔", "安详"], "english": ["peaceful", "calm", "gentle", "serene"]},
            "romantic": {"chinese": ["爱情", "浪漫", "心动", "温柔"], "english": ["love", "romantic", "heartbeat", "tender"]},
            "melancholic": {"chinese": ["忧郁", "沉思", "孤独", "感伤"], "english": ["melancholy", "contemplation", "lonely", "sorrowful"]}
        }

        style_phrases = {
            "pop": {"chinese": "流行旋律", "english": "pop melody"},
            "rock": {"chinese": "摇滚节奏", "english": "rock rhythm"},
            "ballad": {"chinese": "抒情慢歌", "english": "ballad song"},
            "folk": {"chinese": "民谣故事", "english": "folk tale"},
            "electronic": {"chinese": "电子节拍", "english": "electronic beat"},
            "jazz": {"chinese": "爵士风情", "english": "jazz style"},
            "classical": {"chinese": "古典优雅", "english": "classical elegance"}
        }

        mood_word_list = mood_words.get(mood, mood_words["nostalgic"])
        words = mood_word_list.get(language, mood_word_list["chinese"])
        style_phrase = style_phrases.get(style, style_phrases["pop"])

        if language == "chinese":
            return f"{words[0]}的{style_phrase['chinese']}"
        elif language == "english":
            return f"{words[0]} {style_phrase['english']}"
        else:
            return f"{words[0]} {style_phrase['english']}"

    def _format_lyrics(self, lyrics: str, format_style: str = "traditional",
                       rhyme_scheme: str = "none", target_audience: str = "general",
                       language: str = "chinese"):
        """Format and validate lyrics with enhanced controls"""
        lines = lyrics.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('[') and line.endswith(']'):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(line)

        formatted_lyrics = '\n'.join(formatted_lines)
        formatted_lyrics = self._apply_format_style(formatted_lyrics, format_style)
        formatted_lyrics = self._apply_rhyme_scheme(formatted_lyrics, rhyme_scheme, language)
        formatted_lyrics = self._adjust_for_audience(formatted_lyrics, target_audience, language)

        if len(formatted_lyrics) > self.max_lyric_length:
            formatted_lyrics = formatted_lyrics[:self.max_lyric_length] + "..."
            logger.warning(f"⚠️ Lyrics truncated to {self.max_lyric_length} characters")

        return formatted_lyrics

    def _get_fallback_lyrics(self, style: str, mood: str, language: str):
        """Get fallback lyrics when generation fails"""
        fallback = {
            "chinese": """[intro]
轻柔的旋律响起
回忆如潮水般涌来

[verse]
那些年我们一起走过
青春岁月如诗如画
梦想在心中发芽
未来在远方等待

[chorus]
这是我们的青春之歌
唱出心中的梦想
无论走到哪里
永远记得这一刻

[outro]
歌声渐渐远去
回忆永远珍藏""",
            "english": """[intro]
Soft melody begins
Memories come flooding back

[verse]
Those years we walked together
Youthful days like poetry
Dreams sprouting in our hearts
Future waiting in the distance

[chorus]
This is our song of youth
Singing dreams from our hearts
No matter where we go
We'll always remember this moment

[outro]
The song fades away
Memories forever treasured""",
            "mixed": """[intro]
轻柔的旋律 softly plays
回忆 memories flow

[verse]
那些年 those years together
青春 youth like poetry
梦想 dreams in hearts
未来 future awaits

[chorus]
这是我们的 this is our song
唱出梦想 singing dreams
无论走到 wherever we go
永远记得 always remember

[outro]
歌声渐渐 the song fades
回忆珍藏 memories treasured"""
        }
        return fallback.get(language, fallback["chinese"])

    def _apply_format_style(self, lyrics: str, format_style: str) -> str:
        """Apply specific format style to lyrics"""
        if format_style == "minimal":
            lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
            return '\n'.join(lines)
        elif format_style == "detailed":
            lines = lyrics.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip() and not line.startswith('['):
                    formatted_lines.append(f"  {line.strip()}")
                else:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)
        elif format_style == "modern":
            lines = lyrics.split('\n')
            formatted_lines = []
            total = len(lines)
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                if line.startswith('['):
                    if line.strip().lower() == '[outro]' and idx == total - 1:
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append(f"\n{line}")
                else:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)
        else:
            return lyrics

    def _apply_rhyme_scheme(self, lyrics: str, rhyme_scheme: str, language: str) -> str:
        """Apply rhyme scheme to lyrics (simplified)"""
        if rhyme_scheme == "none":
            return lyrics
        lines = lyrics.split('\n')
        lyric_lines = [line for line in lines if line.strip() and not line.startswith('[')]
        if len(lyric_lines) < 2:
            return lyrics
        # Placeholder for more advanced rhyming
        return lyrics

    def _adjust_for_audience(self, lyrics: str, target_audience: str, language: str) -> str:
        """Adjust lyrics content for target audience"""
        if target_audience == "children":
            lines = lyrics.split('\n')
            simplified_lines = []
            for line in lines:
                if line.strip() and not line.startswith('['):
                    simplified_lines.append(line)
                else:
                    simplified_lines.append(line)
            return '\n'.join(simplified_lines)
        elif target_audience == "elderly":
            return lyrics
        else:
            return lyrics

class SongBloomAudioPrompt:
    """
    Node for handling audio prompts for SongBloom
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "target_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("processed_audio", "info")
    FUNCTION = "process_audio_prompt"
    CATEGORY = "SongBloom/Audio"

    def process_audio_prompt(self, audio_file: str = "无 (不使用音频提示)", audio=None, target_duration: float = 10.0):
        """Process audio prompt for SongBloom with comprehensive validation"""
        try:
            # Input validation
            if target_duration <= 0 or target_duration > 30:
                raise AudioProcessingError("Target duration must be between 0 and 30 seconds")

            # 处理音频文件选择
            audio_path = None
            if audio_file and audio_file.strip():
                audio_path = audio_file

            # Load audio from path or use provided audio
            if audio is not None:
                # Validate provided audio
                validate_audio_input(audio)
                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]
                logger.info("Using provided audio tensor")
            elif audio_path and os.path.exists(audio_path):
                # Load from file
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    logger.info(f"Loaded audio from: {audio_path}")
                except Exception as e:
                    raise AudioProcessingError(f"Failed to load audio file: {str(e)}")
            else:
                # 如果没有提供音频，创建静音音频
                logger.info("No audio provided, creating silent audio prompt")
                waveform = torch.zeros(1, int(target_duration * 48000))
                sample_rate = 48000

            # Validate loaded audio
            if waveform.numel() == 0:
                raise AudioProcessingError("Loaded audio is empty")

            # Check for silence
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms < 0.001:
                logger.warning("Audio appears to be very quiet or silent")

            # Resample to 48kHz if needed
            target_sr = 48000
            if sample_rate != target_sr:
                logger.info(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
                sample_rate = target_sr

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                logger.info("Converting stereo to mono")
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # Adjust duration
            target_samples = int(target_duration * sample_rate)
            current_samples = waveform.shape[1]
            current_duration = current_samples / sample_rate

            if current_samples > target_samples:
                # Trim to target duration
                waveform = waveform[:, :target_samples]
                info = f"Audio trimmed from {current_duration:.2f}s to {target_duration}s"
            elif current_samples < target_samples:
                # Pad with zeros
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                info = f"Audio padded from {current_duration:.2f}s to {target_duration}s"
            else:
                info = f"Audio duration is exactly {target_duration}s"

            # Final validation
            final_duration = waveform.shape[1] / sample_rate
            if abs(final_duration - target_duration) > 0.1:
                raise AudioProcessingError(f"Failed to achieve target duration: {final_duration:.2f}s vs {target_duration}s")

            # Prepare output
            processed_audio = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            logger.info(f"Successfully processed audio prompt: {info}")
            return (processed_audio, info)

        except AudioProcessingError as e:
            logger.error(f"Audio processing error: {e}")
            # Return empty audio on error
            empty_audio = {
                "waveform": torch.zeros(1, int(target_duration * 48000)),
                "sample_rate": 48000
            }
            return (empty_audio, f"Audio processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing audio prompt: {e}")
            logger.debug(traceback.format_exc())
            # Return empty audio on error
            empty_audio = {
                "waveform": torch.zeros(1, int(target_duration * 48000)),
                "sample_rate": 48000
            }
            return (empty_audio, f"Unexpected error: {str(e)}")


class SongBloomGenerator:
    """
    Main SongBloom generation node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SONGBLOOM_MODEL",),
                "lyrics": ("STRING", {"multiline": True}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "quality_preset": (["Ultra High", "High", "Balanced", "Fast", "Custom"], {"default": "High"}),
                "low_memory_mode": ("BOOLEAN", {"default": False}),
                "max_memory_retries": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "prompt_max_seconds": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "max_frames": ("INT", {"default": 12000, "min": 1000, "max": 20000, "step": 25}),
                "cfg_coef": ("FLOAT", {
                    "default": 3.0,  # 提高默认值
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 100,  # 提高默认值
                    "min": 10,
                    "max": 500,  # 增加最大值
                    "step": 1
                }),
                "top_k": ("INT", {
                    "default": 100,  # 降低默认值以提高质量
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "use_sampling": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1
                }),
                "dtype": (["float32", "bfloat16"], {
                    "default": "float32"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("generated_audio", "generation_info", "saved_path")
    FUNCTION = "generate_song"
    CATEGORY = "SongBloom/Generation"

    def generate_song(self, model, lyrics: str, audio_prompt=None, quality_preset: str = "High",
                     low_memory_mode: bool = False, max_memory_retries: int = 2, prompt_max_seconds: int = 10,
                     max_frames: int = 12000,
                     cfg_coef: float = 3.0, steps: int = 100, top_k: int = 100, use_sampling: bool = True, seed: int = -1,
                     dtype: str = "float32"):
        """Generate song using SongBloom with quality optimization"""
        try:
            # Hook up ComfyUI progress bar via model callback
            try:
                from comfy.utils import ProgressBar
            except Exception:
                ProgressBar = None
            pb = ProgressBar(steps) if 'ProgressBar' in globals() or ProgressBar else None
            if hasattr(model, 'set_progress_callback') and pb is not None:
                model.set_progress_callback(lambda cur, total: pb.update_absolute(min(cur, total), total))
            # Compute final settings from preset > quality_config > manual
            presets = {
                "Ultra High": {"cfg_coef": 4.0, "steps": 200, "top_k": 50,  "use_sampling": True, "dit_cfg_type": 'h'},
                "High":       {"cfg_coef": 3.0, "steps": 100, "top_k": 100, "use_sampling": True, "dit_cfg_type": 'h'},
                "Balanced":   {"cfg_coef": 2.0, "steps": 75,  "top_k": 150, "use_sampling": True, "dit_cfg_type": 'h'},
                "Fast":       {"cfg_coef": 1.5, "steps": 50,  "top_k": 200, "use_sampling": True, "dit_cfg_type": 'h'},
            }

            if quality_preset != "Custom":
                preset = presets[quality_preset]
                final_cfg_coef = preset["cfg_coef"]
                final_steps = preset["steps"]
                final_top_k = preset["top_k"]
                use_sampling_value = preset["use_sampling"]
                dit_cfg_type = preset["dit_cfg_type"]
                logger.info(f"Using built-in preset: {quality_preset}")
            else:
                final_cfg_coef = cfg_coef
                final_steps = steps
                final_top_k = top_k
                use_sampling_value = use_sampling
                dit_cfg_type = 'h'

                logger.info(f"Using manual settings: CFG={final_cfg_coef}, Steps={final_steps}, Top_k={final_top_k}")

            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                logger.info(f"Set generation seed: {seed}")

            # Prepare audio prompt (optional)
            if audio_prompt is not None:
                prompt_wav = audio_prompt["waveform"]
                if prompt_wav.ndim == 2 and prompt_wav.shape[0] == 1:
                    prompt_wav = prompt_wav.squeeze(0)
                if prompt_wav.ndim == 1:
                    prompt_wav = prompt_wav.unsqueeze(0)
                logger.info(f"Using audio prompt: {prompt_wav.shape}")
            else:
                # Create empty audio prompt when not provided
                prompt_wav = torch.zeros(1, 48000 * max(1, int(prompt_max_seconds)))  # silence
                logger.info("No audio prompt provided, using silence")

            # Clamp prompt length for memory
            try:
                max_prompt_samples = model.sample_rate * max(1, int(prompt_max_seconds))
            except Exception:
                max_prompt_samples = 48000 * max(1, int(prompt_max_seconds))
            if prompt_wav.shape[-1] > max_prompt_samples:
                logger.info(f"Trimming prompt to {prompt_max_seconds}s for memory safety")
                prompt_wav = prompt_wav[:, :max_prompt_samples]

            # Set generation parameters with quality optimization
            logger.info("Setting high-quality generation parameters...")
            # UI优先：覆盖 use_sampling（即使预设/外部配置指定了值）
            use_sampling_value = use_sampling

            model.set_generation_params(
                cfg_coef=final_cfg_coef,
                steps=final_steps,
                top_k=final_top_k,
                use_sampling=use_sampling_value,
                dit_cfg_type=dit_cfg_type,
                max_frames=max_frames
            )

            # Auto-optimize parameters based on GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_free = gpu_memory - gpu_allocated
                logger.info(f"GPU内存状态: {gpu_free:.1f}GB 可用 / {gpu_memory:.1f}GB 总计")
                
                # Auto-adjust parameters based on available memory
                if gpu_free < 2.0:  # Less than 2GB free
                    logger.warning(f"⚠️  GPU内存不足: 仅剩{gpu_free:.1f}GB，自动降低参数")
                    final_cfg_coef = min(final_cfg_coef, 1.5)
                    final_steps = min(final_steps, 30)
                    final_top_k = max(final_top_k, 200)
                    logger.info(f"🔧 自动优化参数: CFG={final_cfg_coef}, Steps={final_steps}, Top_k={final_top_k}")
                elif gpu_free < 4.0:  # Less than 4GB free
                    logger.info(f"💡 GPU内存适中: {gpu_free:.1f}GB，使用平衡参数")
                    final_cfg_coef = min(final_cfg_coef, 2.5)
                    final_steps = min(final_steps, 60)
                    final_top_k = max(final_top_k, 150)
                    logger.info(f"🔧 平衡参数: CFG={final_cfg_coef}, Steps={final_steps}, Top_k={final_top_k}")
                else:
                    logger.info(f"✅ GPU内存充足: {gpu_free:.1f}GB，使用高质量参数")
            
            # Generate with memory monitoring
            print(f"Generating song with lyrics: {lyrics[:100]}...")
            try:
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Ensure input data type matches model dtype (following original project logic)
                torch_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
                logger.info(f"🔧 Generation dtype: {torch_dtype}")
                
                if prompt_wav is not None:
                    logger.info(f"🔧 Input prompt_wav dtype: {prompt_wav.dtype}")
                    if prompt_wav.dtype != torch_dtype:
                        logger.info(f"🔄 Converting prompt_wav from {prompt_wav.dtype} to {torch_dtype}")
                        prompt_wav = prompt_wav.to(dtype=torch_dtype)
                    
                    # Ensure prompt_wav is on the same device as the model
                    if hasattr(model, 'device'):
                        prompt_wav = prompt_wav.to(device=model.device)
                
                # Model dtype conversion is handled in build_from_trainer
                # No need for additional conversion here
                
                # Low-memory autocast
                context = torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
                retries = 0
                last_err = None
                while True:
                    try:
                        with context(dtype=torch.bfloat16 if dtype == 'bfloat16' and torch.cuda.is_available() else None):
                            generated_audio = model.generate(lyrics, prompt_wav)
                        break
                    except RuntimeError as e:
                        if ("out of memory" in str(e).lower() or "allocation" in str(e).lower()) and (low_memory_mode or retries < max_memory_retries):
                            retries += 1
                            # Backoff: reduce steps/top_k/cfg, free cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            steps = max(20, int(steps * 0.6))
                            top_k = min(300, max(50, int(top_k * 1.2)))
                            cfg_coef = min(cfg_coef, 2.0)
                            # Shorten prompt further if needed
                            new_len = max(int(prompt_wav.shape[-1] * 0.8), int(3 * 48000))
                            if prompt_wav.shape[-1] > new_len:
                                prompt_wav = prompt_wav[:, :new_len]
                            logger.warning(f"低内存重试#{retries}: steps={steps}, top_k={top_k}, cfg={cfg_coef}, prompt_len={prompt_wav.shape[-1]/48000:.2f}s")
                            model.set_generation_params(
                                cfg_coef=cfg_coef,
                                steps=steps,
                                top_k=top_k,
                                use_sampling=use_sampling_value,
                                dit_cfg_type='h',
                                max_frames=max_frames
                            )
                            continue
                        last_err = e
                        raise
                
                # Check if generation was successful
                if generated_audio is None or generated_audio.numel() == 0:
                    raise GenerationError("Model returned empty audio")
                
                # Check if audio is silent (likely an error)
                audio_rms = torch.sqrt(torch.mean(generated_audio ** 2)).item()
                if audio_rms < 0.001:
                    logger.warning(f"⚠️  生成的音频可能为静音 (RMS: {audio_rms:.6f})，可能是生成错误")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                    # Clear cache and try with reduced parameters
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning("GPU内存不足，尝试清理缓存...")
                    raise GenerationError(f"GPU内存不足: {str(e)}")
                else:
                    raise

            # Prepare output
            output_audio = {
                "waveform": generated_audio,
                "sample_rate": model.sample_rate
            }

            duration = generated_audio.shape[-1] / model.sample_rate
            generation_info = f"Generated song successfully. Duration: {duration:.2f}s"

            # Audio is not auto-saved, only returned for further processing
            saved_path = "Not saved"

            return (output_audio, generation_info, saved_path)

        except Exception as e:
            error_msg = f"Error generating song: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())

            # Enhanced error handling for different error types
            if "Allocation on device" in str(e) or "CUDA out of memory" in str(e):
                error_msg = f"CUDA内存不足错误: {str(e)}\n建议:\n1. 减少生成步数(steps)\n2. 降低CFG系数\n3. 使用更小的模型\n4. 重启ComfyUI释放内存"
            elif "cuda" in str(e).lower():
                error_msg = f"CUDA设备错误: {str(e)}\n建议:\n1. 检查GPU驱动\n2. 重启ComfyUI\n3. 尝试使用CPU模式"
            
            logger.error(f"详细错误信息: {error_msg}")

            # Return empty audio on error
            empty_audio = {
                "waveform": torch.zeros(1, 1, 48000),  # 1 second of silence
                "sample_rate": 48000
            }
            return (empty_audio, error_msg, "Generation failed")


class SongBloomAudioSaver:
    """
    Node for saving generated audio to various formats with data output ports.
    This node saves audio and provides output ports for connecting to other nodes.
    For audio playback controls, use SongBloomAudioPreview instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {
                    "default": "songbloom_output",
                    "multiline": False
                }),
                "format": (["flac", "wav", "mp3"], {
                    "default": "flac"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "quality": (["high", "medium", "low"], {
                    "default": "high"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_audio"
    CATEGORY = "SongBloom/Audio"
    OUTPUT_NODE = False  # Changed to False since this is for data passing
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when inputs change
        return float("nan")

    def save_audio(self, audio, filename: str, format: str, output_dir: str = "",
                   quality: str = "high"):
        """Save audio to file with enhanced error handling and ComfyUI compatibility"""
        try:
            # Enhanced debug information
            logger.info("=== SongBloomAudioSaver Enhanced Debug Info ===")
            if audio is None:
                logger.error("❌ Audio input is None!")
                return (None, "Error: No audio data provided")

            # Validate audio structure
            if not isinstance(audio, dict):
                logger.error(f"❌ Audio is not a dict, got: {type(audio)}")
                return (None, "Error: Invalid audio format")

            if "waveform" not in audio or "sample_rate" not in audio:
                logger.error(f"❌ Audio missing required keys: {list(audio.keys())}")
                return (None, "Error: Audio missing waveform or sample_rate")

            # Get audio data with validation
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Enhanced audio validation
            if not isinstance(waveform, torch.Tensor):
                logger.error(f"❌ Waveform is not a tensor, got: {type(waveform)}")
                return (None, "Error: Waveform must be a torch.Tensor")

            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                logger.error(f"❌ Invalid sample rate: {sample_rate}")
                return (None, "Error: Invalid sample rate")

            # Debug: Print comprehensive audio info
            logger.info(f"📊 Input audio shape: {waveform.shape}")
            logger.info(f"📊 Input sample rate: {sample_rate}")
            logger.info(f"📊 Audio duration: {waveform.shape[-1] / sample_rate:.3f}s")
            logger.info(f"📊 Audio dtype: {waveform.dtype}")
            logger.info(f"📊 Audio device: {waveform.device}")
            logger.info(f"📊 Audio min/max: {waveform.min().item():.6f}/{waveform.max().item():.6f}")
            logger.info(f"📊 Audio memory usage: {waveform.numel() * waveform.element_size() / 1024:.1f} KB")

            # Check for valid audio data
            if waveform.numel() == 0:
                logger.error("❌ Audio waveform is empty!")
                return (None, "Error: Empty audio waveform")

            duration = waveform.shape[-1] / sample_rate
            if duration < 0.1:
                logger.warning(f"⚠️  Audio is very short: {duration:.3f}s - this may indicate a generation error")
                # Don't fail for short audio, but warn user
            elif duration > 300:  # 5 minutes
                logger.warning(f"⚠️  Audio is very long: {duration:.1f}s - this may cause memory issues")

            # Check for silent audio
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()
            if rms < 0.001:
                logger.warning(f"⚠️  Audio appears to be silent or very quiet (RMS: {rms:.6f})")

            # Check for clipping
            peak = torch.max(torch.abs(waveform)).item()
            if peak > 0.95:
                logger.warning(f"⚠️  Audio may be clipping (peak: {peak:.3f})")

            # Determine output directory
            if not output_dir:
                try:
                    output_dir = folder_paths.get_output_directory()
                except Exception as e:
                    logger.warning(f"⚠️  Could not get ComfyUI output directory: {e}")
                    output_dir = os.path.join(os.getcwd(), "output")
                    logger.info(f"📁 Using fallback output directory: {output_dir}")

            # Ensure output directory exists and is writable
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Test write permission
                test_file = os.path.join(output_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"📁 Output directory verified: {output_dir}")
            except Exception as e:
                logger.error(f"❌ Cannot write to output directory {output_dir}: {e}")
                return (None, f"Error: Cannot write to output directory: {str(e)}")

            # Prepare filename with timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            base_filename = filename.strip()
            
            # Sanitize filename
            import re
            base_filename = re.sub(r'[<>:"/\\|?*]', '_', base_filename)
            
            if not base_filename.endswith(f".{format}"):
                base_filename = f"{base_filename}_{timestamp}.{format}"
            else:
                # Insert timestamp before extension
                name_part = base_filename[:-len(f".{format}")]
                base_filename = f"{name_part}_{timestamp}.{format}"

            file_path = os.path.join(output_dir, base_filename)
            logger.info(f"💾 Saving to: {file_path}")

            # Enhanced audio preprocessing
            save_waveform = waveform.clone()
            
            # Move to CPU first - torchaudio requires CPU tensors
            if save_waveform.is_cuda:
                logger.info("🔄 Moving audio from CUDA to CPU for saving")
                save_waveform = save_waveform.cpu()
            
            # Handle different tensor dimensions
            original_shape = save_waveform.shape
            if save_waveform.ndim == 3:
                # Remove batch dimension if present
                save_waveform = save_waveform.squeeze(0)
                logger.info(f"📊 Squeezed batch dimension: {original_shape} -> {save_waveform.shape}")
            elif save_waveform.ndim == 1:
                # Add channel dimension
                save_waveform = save_waveform.unsqueeze(0)
                logger.info(f"📊 Added channel dimension: {original_shape} -> {save_waveform.shape}")

            # Ensure we have 2D tensor (channels, samples) for torchaudio
            if save_waveform.ndim != 2:
                logger.error(f"❌ Unexpected waveform dimensions: {save_waveform.shape}")
                return (None, "Error: Invalid waveform dimensions")

            # Normalize audio if needed
            if peak > 1.0:
                logger.info(f"📊 Normalizing audio (peak: {peak:.3f})")
                save_waveform = save_waveform / peak

            # Move to CPU if on CUDA device
            if save_waveform.is_cuda:
                logger.info("🔄 Moving audio from CUDA to CPU for saving")
                save_waveform = save_waveform.cpu()
            
            # Ensure tensor is 2D (channels, samples) for torchaudio.save
            if save_waveform.ndim == 1:
                logger.info("🔄 Converting 1D tensor to 2D (adding channel dimension)")
                save_waveform = save_waveform.unsqueeze(0)  # Add channel dimension
            elif save_waveform.ndim == 3:
                logger.info("🔄 Converting 3D tensor to 2D (removing batch dimension)")
                save_waveform = save_waveform.squeeze(0)  # Remove batch dimension
            
            # Convert to float32 if needed (bfloat16 not supported by torchaudio)
            if save_waveform.dtype == torch.bfloat16:
                logger.info("🔄 Converting bfloat16 to float32 for audio saving")
                save_waveform = save_waveform.float()
            
            # Save audio with enhanced error handling
            logger.info(f"💾 Saving audio in {format} format with {quality} quality...")
            
            try:
                if format == "mp3":
                    # For MP3, save as WAV first then convert
                    temp_wav = file_path.replace(".mp3", "_temp.wav")
                    logger.info(f"📁 Creating temporary WAV file: {temp_wav}")
                    torchaudio.save(temp_wav, save_waveform, sample_rate)

                    # Convert to MP3 using ffmpeg if available
                    try:
                        import subprocess
                        bitrate = {"high": "320k", "medium": "192k", "low": "128k"}[quality]
                        cmd = [
                            "ffmpeg", "-i", temp_wav, "-codec:a", "libmp3lame",
                            "-b:a", bitrate, file_path, "-y"
                        ]
                        logger.info(f"🔄 Converting to MP3 with bitrate {bitrate}")
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        os.remove(temp_wav)
                        logger.info(f"✅ Audio converted to MP3: {file_path}")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        # Fallback to WAV if ffmpeg not available
                        fallback_path = file_path.replace(".mp3", ".wav")
                        os.rename(temp_wav, fallback_path)
                        file_path = fallback_path
                        logger.warning(f"⚠️  FFmpeg not available ({e}), saved as WAV instead: {file_path}")
                else:
                    # Save directly for WAV and FLAC
                    torchaudio.save(file_path, save_waveform, sample_rate)
                    logger.info(f"✅ Audio saved directly as {format}: {file_path}")

            except Exception as save_error:
                logger.error(f"❌ Failed to save audio: {save_error}")
                return {
                    "ui": {"audio": []},
                    "result": (None, f"Error saving audio: {str(save_error)}")
                }

            # Verify file was created and check size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ Audio saved successfully: {file_path}")
                logger.info(f"✅ File size: {file_size} bytes ({file_size/1024:.1f} KB)")

                # Enhanced file verification
                try:
                    verify_waveform, verify_sr = torchaudio.load(file_path)
                    verify_duration = verify_waveform.shape[-1] / verify_sr
                    logger.info(f"✅ File verification: {verify_waveform.shape}, {verify_sr}Hz, {verify_duration:.3f}s")
                    
                    # Check if verification matches original
                    if abs(verify_duration - duration) > 0.1:
                        logger.warning(f"⚠️  Duration mismatch: original {duration:.3f}s vs saved {verify_duration:.3f}s")
                    
                except Exception as verify_error:
                    logger.warning(f"⚠️  File verification failed: {verify_error}")
            else:
                logger.error(f"❌ File was not created: {file_path}")
                return {
                    "ui": {"audio": []},
                    "result": (None, "Error: File was not created")
                }

            # Prepare audio output for ComfyUI audio player display
            logger.info("🎵 Preparing audio for ComfyUI player display...")
            try:
                # Create audio output with proper ComfyUI format
                # Use the already processed save_waveform which is on CPU
                output_waveform = save_waveform.clone()

                # Ensure waveform is in the right format for ComfyUI audio player
                # ComfyUI expects (channels, samples) format
                if output_waveform.ndim == 1:
                    output_waveform = output_waveform.unsqueeze(0)
                elif output_waveform.ndim == 3:
                    output_waveform = output_waveform.squeeze(0)

                # Ensure audio is on CPU for ComfyUI display
                if output_waveform.is_cuda:
                    logger.info("🔄 Moving audio to CPU for display")
                    output_waveform = output_waveform.cpu()

                # Create ComfyUI-compatible audio output
                # This format enables ComfyUI to display audio player with duration and controls
                output_audio = {
                    "waveform": output_waveform,
                    "sample_rate": sample_rate
                }
                logger.info(f"✅ Audio prepared for ComfyUI player: {output_waveform.shape}, {sample_rate}Hz, {duration:.2f}s")
            except Exception as audio_error:
                logger.warning(f"⚠️  Failed to prepare audio: {audio_error}")
                output_audio = None

            # Create success message
            success_msg = f"✅ Audio saved successfully!\n"
            success_msg += f"📁 File: {os.path.basename(file_path)}\n"
            success_msg += f"⏱️  Duration: {duration:.2f}s\n"
            success_msg += f"📊 Size: {file_size/1024:.1f} KB\n"
            success_msg += f"🎵 Format: {format.upper()}\n"
            success_msg += f"🎧 Audio player ready"

            logger.info("=== SongBloomAudioSaver Enhanced Complete ===")

            # Prepare UI format for audio player display
            filename = os.path.basename(file_path)
            # Calculate subfolder relative to ComfyUI output directory
            output_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
            file_dir = os.path.dirname(file_path)
            if file_dir.startswith(output_base):
                subfolder = file_dir[len(output_base):].lstrip(os.sep)
            else:
                subfolder = ""
            if subfolder:
                subfolder = subfolder.replace("\\", "/")  # Normalize path separators

            # Also return UI preview so the node面板能显示保存的音频文件
            ui = {
                "audio": [
                    {
                        "filename": os.path.basename(file_path),
                        "subfolder": "",
                        "type": "output"
                    }
                ]
            }

            return {
                "ui": ui,
                "result": (output_audio, file_path)
            }

        except Exception as e:
            error_msg = f"❌ Error saving audio: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())

            # Return error message and empty audio for preview
            try:
                empty_audio = {
                    "waveform": torch.zeros(1, 1000),
                    "sample_rate": 48000
                }
            except:
                empty_audio = None

            # Return error data for error case
            return (empty_audio, error_msg)


class SongBloomAudioPreview:
    """
    Node for displaying audio with advanced player controls and waveform visualization
    Shows duration, play button, waveform, and audio information
    
    This node provides a comprehensive audio preview interface with:
    - Waveform visualization
    - Play/pause/stop controls
    - Volume control
    - Time display and seeking
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "songbloom_preview_audio"
        self.unique_id = str(uuid.uuid4())
        self.num_samples = 1024  # Number of points for downsampled waveform
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview_audio"
    CATEGORY = "SongBloom/Audio"
    OUTPUT_NODE = True

    def process_waveform(self, waveform):
        """Pre-Downsample the waveform to create visualization data"""
        import numpy as np
        
        # Convert to numpy for easier processing
        if isinstance(waveform, np.ndarray):
            audio_data = waveform
        else:
            audio_data = waveform.squeeze(0).cpu().numpy()
        
        # If stereo, convert to mono by averaging channels
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Ensure 1D array
        audio_data = audio_data.flatten()
        
        # Calculate the block size for downsampling
        total_samples = len(audio_data)
        block_size = max(1, total_samples // self.num_samples)
        
        # Create downsampled data
        peaks = []
        for i in range(self.num_samples):
            block_start = min(i * block_size, total_samples - 1)
            block_end = min(block_start + block_size, total_samples)
            
            if block_start >= block_end:
                # We've reached the end of the audio data
                break
                
            # Get the average absolute amplitude in this block
            block = audio_data[block_start:block_end]
            peak = np.mean(np.abs(block))
            
            # Normalize and add minimum height for visibility
            peaks.append(float(peak))  # Ensure it's a standard Python float
            
        # Pad with zeros if we didn't get enough points
        while len(peaks) < self.num_samples:
            peaks.append(0.1)  # Minimum height

        return peaks

    def preview_audio(self, audio):
        """Display audio with advanced player controls and waveform visualization"""
        try:
            if audio is None or not isinstance(audio, dict):
                return {"ui": {"audio": []}}

            if "waveform" not in audio or "sample_rate" not in audio:
                return {"ui": {"audio": []}}

            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Ensure waveform is 2D (channels, samples)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 3:
                waveform = waveform.squeeze(0)

            # No volume adjustment needed
            
            # Move to CPU if on CUDA device
            if waveform.is_cuda:
                logger.info("🔄 Moving audio from CUDA to CPU for saving")
                waveform = waveform.cpu()
            
            # Ensure tensor is 2D (channels, samples) for torchaudio.save
            if waveform.ndim == 1:
                logger.info("🔄 Converting 1D tensor to 2D (adding channel dimension)")
                waveform = waveform.unsqueeze(0)
            
            # Convert to float32 if needed (bfloat16 not supported by torchaudio)
            if waveform.dtype == torch.bfloat16:
                logger.info("🔄 Converting bfloat16 to float32 for audio saving")
                waveform = waveform.float()

            # Generate unique filename
            import time
            filename = f"SongBloom_Audio_{int(time.time())}.wav"
            full_path = os.path.join(self.output_dir, filename)

            # Ensure directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save audio file
            torchaudio.save(full_path, waveform, sample_rate)
            
            # Process waveform data for visualization
            waveform_data = self.process_waveform(waveform)
            
            # Calculate duration safely
            if waveform.ndim >= 2:
                duration = waveform.shape[-1] / sample_rate
            else:
                duration = len(waveform) / sample_rate
            
            # Convert waveform data to base64 string
            import json
            import base64
            waveform_json = json.dumps({"waveform": waveform_data, "duration": duration})
            waveform_base64 = base64.b64encode(waveform_json.encode('utf-8')).decode('utf-8')

            # Return UI format with waveform data
            return {"ui": {"audio": [{
                "filename": filename, 
                "subfolder": "", 
                "type": self.type,
                "waveform_data": waveform_base64,
                "node_id": self.unique_id
            }]}}

        except Exception as e:
            logger.error(f"Error in SongBloom Audio Preview: {e}")
            return {"ui": {"audio": []}}


class SongBloomAudioCropper:
    """
    Node for cropping a segment of audio by start/end time in seconds.
    - Accepts ComfyUI AUDIO dict {waveform: Tensor, sample_rate: int}
    - Returns cropped AUDIO in the same format
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e6, "step": 0.01}),
                "end_sec": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1e6, "step": 0.01}),
            },
            "optional": {
                "pad_to_end": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("cropped_audio", "info")
    FUNCTION = "crop"
    CATEGORY = "SongBloom/Audio"

    def crop(self, audio, start_sec: float, end_sec: float, pad_to_end: bool = False):
        try:
            # Validate input structure
            validate_audio_input(audio)
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if start_sec < 0 or end_sec < 0:
                raise AudioProcessingError("start_sec 和 end_sec 不能为负数")
            if end_sec <= start_sec:
                raise AudioProcessingError("end_sec 必须大于 start_sec")

            # Normalize tensor shape to (channels, samples)
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            total_samples = waveform.shape[-1]
            total_duration = total_samples / sample_rate

            # Clamp times within available duration
            start_sec_clamped = max(0.0, min(float(start_sec), total_duration))
            end_sec_clamped = max(0.0, min(float(end_sec), total_duration))
            if end_sec_clamped <= start_sec_clamped:
                # If clamped values invert, make a minimal 0.01s window if padding allowed
                if pad_to_end:
                    end_sec_clamped = min(total_duration, start_sec_clamped + 0.01)
                else:
                    raise AudioProcessingError("裁剪时间范围无效或超出音频长度")

            start_idx = int(round(start_sec_clamped * sample_rate))
            end_idx = int(round(end_sec_clamped * sample_rate))

            cropped = waveform[:, start_idx:end_idx]

            # Guard: handle empty selection
            if cropped.numel() == 0 or cropped.shape[-1] == 0:
                if pad_to_end:
                    # create minimal-length silence of 0.01s
                    min_len = max(1, int(round(0.01 * sample_rate)))
                    cropped = torch.zeros(waveform.shape[0], min_len, dtype=waveform.dtype, device=waveform.device)
                else:
                    raise AudioProcessingError("裁剪结果为空，请检查起止时间是否有效")

            # Optional padding if requested and end exceeds original
            target_len = int(round((end_sec - start_sec) * sample_rate))
            if pad_to_end and cropped.shape[-1] < target_len:
                pad_amount = target_len - cropped.shape[-1]
                cropped = torch.nn.functional.pad(cropped, (0, pad_amount))

            # Ensure (channels, samples)
            if cropped.ndim == 1:
                cropped = cropped.unsqueeze(0)
            elif cropped.ndim == 3:
                cropped = cropped.squeeze(0)
            # Make contiguous to be safe for downstream operations
            cropped = cropped.contiguous()

            info = (
                f"原时长 {total_duration:.2f}s, 裁剪: [{start_sec_clamped:.2f}s ~ {end_sec_clamped:.2f}s] "
                f"=> 输出 {cropped.shape[-1] / sample_rate:.2f}s"
            )

            return ({"waveform": cropped, "sample_rate": sample_rate}, info)

        except Exception as e:
            return (audio, f"裁剪失败: {str(e)}")


class SongBloomAudioEnsure2D:
    """
    Utility node: ensure AUDIO waveform is 2D (channels, samples).
    Useful when downstream nodes expect 2D and some upstream produced 1D.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "fix"
    CATEGORY = "SongBloom/Audio"

    def fix(self, audio):
        try:
            # Be tolerant to inputs that don't match expected structure
            if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
                raise AudioProcessingError("AUDIO 格式不正确，应包含 waveform 与 sample_rate")

            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"]) if not isinstance(audio["sample_rate"], int) else audio["sample_rate"]

            # Convert non-tensor inputs to tensor
            if not isinstance(waveform, torch.Tensor):
                try:
                    import numpy as _np
                    if isinstance(waveform, _np.ndarray):
                        waveform = torch.from_numpy(waveform)
                    elif isinstance(waveform, (list, tuple)):
                        waveform = torch.tensor(waveform)
                    else:
                        # scalar or unsupported -> make minimal audio
                        waveform = torch.tensor([float(waveform)])
                except Exception:
                    waveform = torch.tensor([0.0])

            original_shape = tuple(waveform.shape)

            # Normalize to 2D: (channels, samples)
            if waveform.ndim == 0:
                waveform = waveform.reshape(1, 1)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim >= 3:
                # Common cases: (1, C, S) or (B, C, S) -> keep first channel and drop batch
                # Try to reduce to (C, S)
                while waveform.ndim > 2:
                    if waveform.shape[0] == 1:
                        waveform = waveform.squeeze(0)
                    else:
                        # collapse leading dims into channel dim
                        new_c = int(waveform.shape[0])
                        waveform = waveform.reshape(new_c, -1)
                        break

            # Guarantee shape semantics (channels, samples)
            if waveform.shape[0] > 2 and waveform.ndim == 2:
                # Likely (samples, channels) -> transpose to (channels, samples)
                if waveform.shape[1] in (1, 2):
                    waveform = waveform.mT

            # Ensure contiguous tensor on CPU for downstream compatibility
            if waveform.is_cuda:
                waveform = waveform.cpu()
            waveform = waveform.contiguous()

            info = f"waveform {original_shape} -> {tuple(waveform.shape)}"
            return ({"waveform": waveform, "sample_rate": sample_rate}, info)
        except Exception as e:
            return (audio, f"Ensure2D 失败: {str(e)}")

class SongBloomQualityConfig:
    """
    Node for high-quality audio generation configuration
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_preset": (["Ultra High", "High", "Balanced", "Fast"], {
                    "default": "High"
                }),
            },
            "optional": {
                "custom_cfg_coef": ("FLOAT", {
                    "default": -1.0,  # -1 means use preset
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "custom_steps": ("INT", {
                    "default": -1,  # -1 means use preset
                    "min": 10,
                    "max": 500,
                    "step": 1
                }),
                "custom_top_k": ("INT", {
                    "default": -1,  # -1 means use preset
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "enable_deterministic": ("BOOLEAN", {
                    "default": False,
                    "label_on": "确定性生成",
                    "label_off": "随机生成"
                }),
            }
        }

    RETURN_TYPES = ("SONGBLOOM_QUALITY_CONFIG",)
    RETURN_NAMES = ("quality_config",)
    FUNCTION = "create_quality_config"
    CATEGORY = "SongBloom/Quality"

    def create_quality_config(self, quality_preset: str, custom_cfg_coef: float = -1.0,
                            custom_steps: int = -1, custom_top_k: int = -1,
                            enable_deterministic: bool = False):
        """Create quality configuration based on preset or custom settings"""

        # Define quality presets
        presets = {
            "Ultra High": {
                "cfg_coef": 4.0,
                "steps": 200,
                "top_k": 50,
                "description": "最高质量，生成时间最长"
            },
            "High": {
                "cfg_coef": 3.0,
                "steps": 100,
                "top_k": 100,
                "description": "高质量，推荐设置"
            },
            "Balanced": {
                "cfg_coef": 2.0,
                "steps": 75,
                "top_k": 150,
                "description": "质量与速度平衡"
            },
            "Fast": {
                "cfg_coef": 1.5,
                "steps": 50,
                "top_k": 200,
                "description": "快速生成，质量一般"
            }
        }

        # Get preset configuration
        preset_config = presets[quality_preset]

        # Apply custom overrides if provided
        final_config = {
            "cfg_coef": custom_cfg_coef if custom_cfg_coef > 0 else preset_config["cfg_coef"],
            "steps": custom_steps if custom_steps > 0 else preset_config["steps"],
            "top_k": custom_top_k if custom_top_k > 0 else preset_config["top_k"],
            "use_sampling": not enable_deterministic,
            "dit_cfg_type": 'h',
            "preset_name": quality_preset,
            "description": preset_config["description"]
        }

        logger.info(f"Quality Config - {quality_preset}: CFG={final_config['cfg_coef']}, "
                   f"Steps={final_config['steps']}, Top_k={final_config['top_k']}")

        return (final_config,)


class SongBloomAdvancedConfig:
    """
    Node for advanced SongBloom generation configuration
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_coef": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 1
                }),
                "top_k": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "use_sampling": ("BOOLEAN", {"default": True}),
                "dit_cfg_type": (["h", "l"], {"default": "h"}),
            },
            "optional": {
                "max_frames": ("INT", {
                    "default": 3750,  # 150s * 25fps
                    "min": 250,
                    "max": 10000,
                    "step": 25
                }),
            }
        }

    RETURN_TYPES = ("SONGBLOOM_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "SongBloom/Utils"

    def create_config(self, cfg_coef: float, steps: int, top_k: int,
                     use_sampling: bool, dit_cfg_type: str, max_frames: int = 3750):
        """Create advanced configuration for SongBloom"""
        config = {
            "cfg_coef": cfg_coef,
            "steps": steps,
            "top_k": top_k,
            "use_sampling": use_sampling,
            "dit_cfg_type": dit_cfg_type,
            "max_frames": max_frames
        }
        return (config,)


class SongBloomLyricValidator:
    """
    Node for validating and analyzing lyric format
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("validation_report", "suggestions", "is_valid")
    FUNCTION = "validate_lyrics"
    CATEGORY = "SongBloom/Utils"

    def validate_lyrics(self, lyrics: str):
        """Validate lyric format and provide detailed feedback"""
        try:
            issues = []
            suggestions = []

            # Check for structure tags
            structure_tags = ['[intro]', '[verse]', '[chorus]', '[bridge]', '[inst]', '[outro]']
            found_tags = []
            for tag in structure_tags:
                if tag in lyrics:
                    found_tags.append(tag)

            if not found_tags:
                issues.append("No structure tags found")
                suggestions.append("Add structure tags like [verse], [chorus], [intro], [outro]")

            # Check for proper separators
            sections = lyrics.split(',')
            if len(sections) == 1 and len(found_tags) > 1:
                issues.append("Missing section separators")
                suggestions.append("Use ',' to separate different sections")

            # Check vocal sections for sentence separators
            for i, section in enumerate(sections):
                section = section.strip()
                if any(tag in section for tag in ['[verse]', '[chorus]', '[bridge]']):
                    # This is a vocal section
                    if '.' not in section:
                        issues.append(f"Section {i+1} missing sentence separators")
                        suggestions.append(f"Add '.' to separate sentences in vocal sections")

            # Check for balanced structure
            intro_count = lyrics.count('[intro]')
            outro_count = lyrics.count('[outro]')
            verse_count = lyrics.count('[verse]')
            chorus_count = lyrics.count('[chorus]')

            if intro_count == 0:
                suggestions.append("Consider adding an [intro] section")
            if outro_count == 0:
                suggestions.append("Consider adding an [outro] section")
            if verse_count == 0:
                issues.append("No [verse] sections found")
                suggestions.append("Add at least one [verse] section")
            if chorus_count == 0:
                suggestions.append("Consider adding a [chorus] section for better song structure")

            # Estimate duration
            total_tags = sum(lyrics.count(tag) for tag in structure_tags)
            estimated_duration = total_tags * 1.0  # Rough estimate: 1 second per tag

            # Generate report
            is_valid = len(issues) == 0

            report_lines = [
                f"=== SongBloom Lyric Validation Report ===",
                f"Structure tags found: {', '.join(found_tags)}",
                f"Estimated duration: ~{estimated_duration:.0f} seconds",
                f"Sections count: {len(sections)}",
                ""
            ]

            if issues:
                report_lines.append("❌ Issues found:")
                for issue in issues:
                    report_lines.append(f"  - {issue}")
                report_lines.append("")
            else:
                report_lines.append("✅ No critical issues found")
                report_lines.append("")

            if suggestions:
                report_lines.append("💡 Suggestions:")
                for suggestion in suggestions:
                    report_lines.append(f"  - {suggestion}")

            validation_report = "\n".join(report_lines)
            suggestions_text = "\n".join(suggestions) if suggestions else "No suggestions"

            return (validation_report, suggestions_text, is_valid)

        except Exception as e:
            error_msg = f"Error validating lyrics: {str(e)}"
            return (error_msg, "", False)


class SongBloomAudioAnalyzer:
    """
    Node for analyzing audio properties and compatibility
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("analysis_report", "is_suitable")
    FUNCTION = "analyze_audio"
    CATEGORY = "SongBloom/Utils"

    def analyze_audio(self, audio):
        """Analyze audio for SongBloom compatibility"""
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Calculate properties
            duration = waveform.shape[-1] / sample_rate
            channels = waveform.shape[0] if waveform.ndim > 1 else 1

            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(waveform ** 2)).item()

            # Calculate peak amplitude
            peak = torch.max(torch.abs(waveform)).item()

            # Check for silence
            silence_threshold = 0.001
            is_silent = rms < silence_threshold

            # Suitability checks
            issues = []
            if duration < 5.0:
                issues.append("Audio too short (< 5 seconds)")
            if duration > 30.0:
                issues.append("Audio too long (> 30 seconds), will be trimmed")
            if sample_rate not in [44100, 48000]:
                issues.append(f"Non-standard sample rate ({sample_rate}Hz), will be resampled")
            if is_silent:
                issues.append("Audio appears to be silent or very quiet")
            if peak > 0.95:
                issues.append("Audio may be clipping (peak > 0.95)")

            is_suitable = len(issues) == 0 or all("will be" in issue for issue in issues)

            # Generate report
            report_lines = [
                "=== Audio Analysis Report ===",
                f"Duration: {duration:.2f} seconds",
                f"Sample Rate: {sample_rate} Hz",
                f"Channels: {channels}",
                f"RMS Energy: {rms:.4f}",
                f"Peak Amplitude: {peak:.4f}",
                ""
            ]

            if issues:
                report_lines.append("⚠️ Issues/Notes:")
                for issue in issues:
                    report_lines.append(f"  - {issue}")
            else:
                report_lines.append("✅ Audio is suitable for SongBloom")

            analysis_report = "\n".join(report_lines)

            return (analysis_report, is_suitable)

        except Exception as e:
            error_msg = f"Error analyzing audio: {str(e)}"
            return (error_msg, False)


class SongBloomBatchProcessor:
    """
    Node for batch processing multiple lyrics with the same model and prompt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SONGBLOOM_MODEL",),
                "lyrics_list": ("STRING", {
                    "multiline": True,
                    "default": "# Enter multiple lyrics separated by '---'\n# Example:\nLyrics 1 here\n---\nLyrics 2 here\n---\nLyrics 3 here"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "cfg_coef": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 1
                }),
                "base_filename": ("STRING", {
                    "default": "batch_song",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("batch_report", "songs_generated")
    FUNCTION = "process_batch"
    CATEGORY = "SongBloom/Utils"

    def process_batch(self, model, lyrics_list: str, batch_size: int, audio_prompt=None,
                     cfg_coef: float = 1.5, steps: int = 50, base_filename: str = "batch_song"):
        """Process multiple lyrics in batch"""
        try:
            # Parse lyrics list
            lyrics_entries = [entry.strip() for entry in lyrics_list.split('---') if entry.strip()]
            lyrics_entries = [entry for entry in lyrics_entries if not entry.startswith('#')]

            if not lyrics_entries:
                return ("No valid lyrics found in batch list", 0)

            # Set generation parameters
            model.set_generation_params(
                cfg_coef=cfg_coef,
                steps=steps,
                use_sampling=True,
                dit_cfg_type='h'
            )

            # Prepare audio prompt (optional)
            if audio_prompt is not None:
                prompt_wav = audio_prompt["waveform"]
                if prompt_wav.ndim == 2 and prompt_wav.shape[0] == 1:
                    prompt_wav = prompt_wav.squeeze(0)
                if prompt_wav.ndim == 1:
                    prompt_wav = prompt_wav.unsqueeze(0)
                logger.info(f"Using audio prompt for batch: {prompt_wav.shape}")
            else:
                # Create empty audio prompt when not provided
                prompt_wav = torch.zeros(1, 48000 * 10)  # 10 seconds of silence at 48kHz
                logger.info("No audio prompt provided for batch, using silence")

            # Process in batches
            generated_count = 0
            output_dir = folder_paths.get_output_directory()

            report_lines = [
                f"=== Batch Processing Report ===",
                f"Total lyrics entries: {len(lyrics_entries)}",
                f"Batch size: {batch_size}",
                f"Output directory: {output_dir}",
                ""
            ]

            for i, lyrics in enumerate(lyrics_entries):
                try:
                    print(f"Processing batch item {i+1}/{len(lyrics_entries)}")

                    # Generate song
                    generated_audio = model.generate(lyrics, prompt_wav)

                    # Save audio
                    filename = f"{base_filename}_{i+1:03d}.flac"
                    file_path = os.path.join(output_dir, filename)

                    # Ensure correct shape for saving
                    save_audio = generated_audio
                    if save_audio.ndim == 3:
                        save_audio = save_audio.squeeze(0)
                    
                    # Move to CPU if on CUDA device
                    if save_audio.is_cuda:
                        logger.info("🔄 Moving audio from CUDA to CPU for saving")
                        save_audio = save_audio.cpu()
                    
                    # Ensure tensor is 2D (channels, samples) for torchaudio.save
                    if save_audio.ndim == 1:
                        logger.info("🔄 Converting 1D tensor to 2D (adding channel dimension)")
                        save_audio = save_audio.unsqueeze(0)
                    
                    # Convert to float32 if needed (bfloat16 not supported by torchaudio)
                    if save_audio.dtype == torch.bfloat16:
                        logger.info("🔄 Converting bfloat16 to float32 for audio saving")
                        save_audio = save_audio.float()

                    torchaudio.save(file_path, save_audio, model.sample_rate)

                    generated_count += 1
                    report_lines.append(f"✅ Generated: {filename}")

                except Exception as e:
                    report_lines.append(f"❌ Failed item {i+1}: {str(e)}")

            report_lines.append(f"\nSuccessfully generated: {generated_count}/{len(lyrics_entries)} songs")
            batch_report = "\n".join(report_lines)

            return (batch_report, generated_count)

        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            print(error_msg)
            return (error_msg, 0)


class SongBloomDeepSeekLyricFormatter:
    """
    保持节点名与显示名不变，仅对齐参考项目 DeepSeek V3 的实现与 API 调用方式（OpenAI SDK）。
    - 输入与本项目原设计一致：lyrics + 可选 api_key/model/system_prompt。
    - 内部将 DeepSeek-V3 映射为 deepseek-chat。
    - 输出保持原样：formatted_lyrics + raw_response 便于调试。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["DeepSeek-V3"], {"default": "DeepSeek-V3"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("formatted_lyrics", "raw_response")
    FUNCTION = "format_lyrics"
    CATEGORY = "SongBloom/Text"

    def format_lyrics(self, lyrics: str, api_key: str = "", model: str = "DeepSeek-V3", system_prompt: str = "", seed: int = 0):
        # 解析 API Key：优先 节点输入 > 环境变量 > DeepseekAP-config.json > config.yaml: deepseek.api_key
        def _load_api_from_cfg():
            try:
                import yaml
                cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                return (data.get('deepseek') or {}).get('api_key', '')
            except Exception:
                return ''

        def _load_api_from_json():
            try:
                json_path = os.path.join(os.path.dirname(__file__), "DeepseekAP-config.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        j = json.load(jf) or {}
                    return j.get('api_key', '')
            except Exception:
                return ''
            return ''

        if (api_key or "").strip():
            API_KEY = api_key.strip()
        elif os.getenv("DEEPSEEK_API_KEY") is not None:
            API_KEY = os.getenv("DEEPSEEK_API_KEY")
        else:
            json_key = _load_api_from_json()
            cfg_key = json_key or _load_api_from_cfg()
            if cfg_key:
                API_KEY = cfg_key
            else:
                return (lyrics, "DeepSeek API Key 未设置。请在节点 api_key 中填写，或设置环境变量 DEEPSEEK_API_KEY，或在 config.yaml 的 deepseek.api_key 中配置。")

        # 模型名映射
        model_alias = (model or "").strip()
        model_final = "deepseek-chat" if model_alias.lower() in ["deepseek-v3", "deepseek_v3", "deepseek v3", "deepseek-chat", "deepseek_chat"] else model_alias or "deepseek-chat"

        # SDK 调用与对齐参考项目
        try:
            client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
            messages = []
            if (system_prompt or "").strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": lyrics})

            completion = client.chat.completions.create(
                model=model_final,
                messages=messages,
                stream=False,
            )

            text = completion.choices[0].message.content if completion and completion.choices else ""
            import json as _json
            return (text, _json.dumps(completion.to_dict_recursive() if hasattr(completion, 'to_dict_recursive') else {}, ensure_ascii=False))
        except Exception as e:
            return (lyrics, f"DeepSeek 调用失败: {str(e)}")

class SongBloomDeepSeekV3:
    pass

# (Removed) Chunked generation implementation was deprecated and is now fully removed.
class SongBloomChunkedGenerator:
    """
    Generate long audio in chunks to降低峰值显存占用。
    - 将歌词按逗号","或换行分段（段内建议包含结构标签如 [verse]/[chorus]）。
    - 每段单独调用 model.generate，然后在时间轴上拼接；相邻段之间可使用上一段尾部作为提示音，实现平滑过渡。
    注意：该策略更稳，但段间上下文会变弱，属于内存优先方案。
    """

    pass

# Node mappings
NODE_CLASS_MAPPINGS = {
    "SongBloomModelLoader": SongBloomModelLoader,
    "SongBloomLyricProcessor": SongBloomLyricProcessor,
    "SongBloomLyricGenerator": SongBloomLyricGenerator,
    "SongBloomAudioCropper": SongBloomAudioCropper,
    "SongBloomAudioEnsure2D": SongBloomAudioEnsure2D,
    "SongBloomAudioPrompt": SongBloomAudioPrompt,
    "SongBloomGenerator": SongBloomGenerator,
    "SongBloomDeepSeekLyricFormatter": SongBloomDeepSeekLyricFormatter,
    "SongBloomAudioSaver": SongBloomAudioSaver,
    "SongBloomAudioPreview": SongBloomAudioPreview,
    # "SongBloomQualityConfig": SongBloomQualityConfig,  # removed per simplification
    "SongBloomAdvancedConfig": SongBloomAdvancedConfig,
    "SongBloomLyricValidator": SongBloomLyricValidator,
    "SongBloomAudioAnalyzer": SongBloomAudioAnalyzer,
    "SongBloomBatchProcessor": SongBloomBatchProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongBloomModelLoader": "SongBloom Model Loader",
    "SongBloomLyricProcessor": "SongBloom Lyric Processor",
    "SongBloomLyricGenerator": "SongBloom Lyric Generator",
    "SongBloomAudioCropper": "SongBloom Audio Cropper",
    "SongBloomAudioEnsure2D": "SongBloom Audio Ensure2D",
    "SongBloomAudioPrompt": "SongBloom Audio Prompt",
    "SongBloomGenerator": "SongBloom Generator",
    "SongBloomDeepSeekLyricFormatter": "SongBloom DeepSeek Lyric Formatter",
    "SongBloomAudioSaver": "SongBloom Audio Saver",
    "SongBloomAudioPreview": "SongBloom Audio Preview",
    # "SongBloomQualityConfig": "SongBloom Quality Config",
    "SongBloomAdvancedConfig": "SongBloom Advanced Config",
    "SongBloomLyricValidator": "SongBloom Lyric Validator",
    "SongBloomAudioAnalyzer": "SongBloom Audio Analyzer",
    "SongBloomBatchProcessor": "SongBloom Batch Processor",
}
