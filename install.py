#!/usr/bin/env python3
"""
Installation script for ComfyUI SongBloom Plugin

This script handles the installation of dependencies and setup for the SongBloom plugin.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_torch():
    """Check if PyTorch is installed and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - CPU-only mode")
        
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_comfyui():
    """Check if ComfyUI is available"""
    try:
        # Try to import folder_paths which is a ComfyUI module
        spec = importlib.util.find_spec("folder_paths")
        if spec is not None:
            print("✅ ComfyUI detected")
            return True
        else:
            print("❌ ComfyUI not found in Python path")
            return False
    except Exception as e:
        print(f"❌ Error checking ComfyUI: {e}")
        return False

def setup_models_directory():
    """Create models directory structure"""
    try:
        # Try to get ComfyUI models directory
        import folder_paths
        models_dir = Path(folder_paths.models_dir) / "SongBloom"
        models_dir.mkdir(exist_ok=True)
        print(f"✅ Models directory created: {models_dir}")
        return True
    except Exception as e:
        print(f"⚠️  Could not setup models directory: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        "torch",
        "torchaudio", 
        "transformers",
        "omegaconf",
        "huggingface_hub",
        "einops",
        "lightning"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_songbloom_import():
    """Test if SongBloom components can be imported"""
    try:
        # Add SongBloom to path
        songbloom_path = Path(__file__).parent / "SongBloom"
        if songbloom_path.exists():
            sys.path.insert(0, str(songbloom_path))
            
            # Test imports
            from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
            from SongBloom.g2p.lyric_common import key2processor
            
            print("✅ SongBloom components imported successfully")
            return True
        else:
            print("❌ SongBloom directory not found")
            return False
    except ImportError as e:
        print(f"❌ Failed to import SongBloom components: {e}")
        return False

def main():
    """Main installation function"""
    print("🎵 ComfyUI SongBloom Plugin Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check ComfyUI
    if not check_comfyui():
        print("\n⚠️  ComfyUI not detected. Make sure you're running this from within ComfyUI environment.")
        print("   If ComfyUI is installed elsewhere, you may need to adjust your Python path.")
    
    # Check PyTorch
    if not check_torch():
        print("\n❌ PyTorch is required. Please install PyTorch first:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        sys.exit(1)
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        print("\n❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Some dependencies are missing. Please check the installation.")
        sys.exit(1)
    
    # Setup models directory
    print("\n📁 Setting up models directory...")
    setup_models_directory()
    
    # Test SongBloom import
    print("\n🧪 Testing SongBloom components...")
    if not test_songbloom_import():
        print("\n⚠️  SongBloom components test failed. The plugin may still work if dependencies are correct.")
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Look for SongBloom nodes in the node menu")
    print("3. Check the example workflows in example_workflows/")
    print("4. Models will be downloaded automatically on first use")
    print("\nFor support, check the README.md file")

if __name__ == "__main__":
    main()
