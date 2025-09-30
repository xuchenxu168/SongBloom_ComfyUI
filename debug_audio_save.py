#!/usr/bin/env python3
"""
调试音频保存功能

检查SongBloomAudioSaver节点的保存功能是否正常工作
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def create_test_audio(duration=5.0):
    """创建测试音频数据"""
    sample_rate = 48000
    frequency = 440  # A4音符
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

def test_audio_save_function():
    """测试音频保存功能"""
    print("🔧 测试音频保存功能...")
    
    def mock_save_audio(audio, filename, format, output_dir="", enable_preview=True):
        """模拟音频保存逻辑"""
        try:
            # 检查输入
            if audio is None:
                return "错误: 没有音频数据", None
            
            print(f"📊 输入音频信息:")
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            print(f"  - 波形形状: {waveform.shape}")
            print(f"  - 采样率: {sample_rate}")
            print(f"  - 时长: {waveform.shape[-1] / sample_rate:.2f}秒")
            print(f"  - 数据类型: {waveform.dtype}")
            print(f"  - 设备: {waveform.device}")
            
            # 检查音频数据是否有效
            if waveform.numel() == 0:
                return "错误: 音频数据为空", None
            
            if waveform.shape[-1] < 1000:  # 少于1000个样本
                return f"警告: 音频太短 ({waveform.shape[-1]}个样本)", None
            
            # 确定输出目录
            if not output_dir:
                output_dir = str(current_dir / "test_output")
            
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 输出目录: {output_dir}")
            
            # 准备文件名
            import time
            timestamp = int(time.time())
            if not filename.endswith(f".{format}"):
                filename = f"{filename}_{timestamp}.{format}"
            
            file_path = os.path.join(output_dir, filename)
            print(f"💾 保存路径: {file_path}")
            
            # 准备保存的音频数据
            save_waveform = waveform.clone()
            if save_waveform.ndim == 3:
                save_waveform = save_waveform.squeeze(0)
            
            print(f"📊 保存音频形状: {save_waveform.shape}")
            
            # 保存音频
            if format == "wav":
                torchaudio.save(file_path, save_waveform, sample_rate)
            elif format == "flac":
                torchaudio.save(file_path, save_waveform, sample_rate)
            else:
                return f"错误: 不支持的格式 {format}", None
            
            # 验证文件是否创建
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ 文件保存成功: {file_size} 字节")
                
                # 验证文件内容
                try:
                    loaded_waveform, loaded_sr = torchaudio.load(file_path)
                    print(f"✅ 文件验证成功: {loaded_waveform.shape}, {loaded_sr}Hz")
                except Exception as e:
                    print(f"⚠️  文件验证失败: {e}")
                
                return file_path, {"waveform": save_waveform, "sample_rate": sample_rate}
            else:
                return "错误: 文件未创建", None
                
        except Exception as e:
            print(f"❌ 保存过程异常: {e}")
            import traceback
            traceback.print_exc()
            return f"错误: {str(e)}", None
    
    # 测试不同的音频数据
    test_cases = [
        ("正常音频", create_test_audio(5.0)),
        ("短音频", create_test_audio(0.1)),
        ("长音频", create_test_audio(10.0)),
    ]
    
    results = []
    for desc, audio in test_cases:
        print(f"\n--- 测试: {desc} ---")
        result, preview = mock_save_audio(audio, f"test_{desc}", "wav", "", True)
        
        if "错误" not in result and "警告" not in result:
            print(f"✅ {desc}: 保存成功")
            results.append(True)
        else:
            print(f"❌ {desc}: {result}")
            results.append(False)
    
    return all(results)

def test_audio_data_flow():
    """测试音频数据流"""
    print("\n🔄 测试音频数据流...")
    
    # 模拟从生成器到保存器的数据流
    def simulate_data_flow():
        print("1. 模拟生成器输出...")
        
        # 创建类似生成器输出的音频数据
        generated_audio = torch.randn(1, 1, 48000 * 5)  # 5秒音频
        output_audio = {
            "waveform": generated_audio,
            "sample_rate": 48000
        }
        
        print(f"   生成器输出: {generated_audio.shape}")
        
        print("2. 传递到保存器...")
        
        # 检查数据是否正确传递
        received_audio = output_audio
        waveform = received_audio["waveform"]
        sample_rate = received_audio["sample_rate"]
        
        print(f"   保存器接收: {waveform.shape}, {sample_rate}Hz")
        
        # 检查数据完整性
        if waveform.numel() > 0 and sample_rate > 0:
            duration = waveform.shape[-1] / sample_rate
            print(f"   音频时长: {duration:.2f}秒")
            
            if duration > 0.1:  # 至少0.1秒
                print("✅ 数据流正常")
                return True
            else:
                print("❌ 音频太短")
                return False
        else:
            print("❌ 数据无效")
            return False
    
    return simulate_data_flow()

def check_output_directory():
    """检查输出目录"""
    print("\n📁 检查输出目录...")
    
    # 检查ComfyUI输出目录
    possible_dirs = [
        "D:\\Ken_ComfyUI_312\\ComfyUI\\output",
        "d:\\audio",
        str(current_dir / "test_output")
    ]
    
    for dir_path in possible_dirs:
        print(f"检查目录: {dir_path}")
        
        if os.path.exists(dir_path):
            print(f"  ✅ 目录存在")
            
            # 检查权限
            if os.access(dir_path, os.W_OK):
                print(f"  ✅ 可写权限")
                
                # 列出现有文件
                try:
                    files = os.listdir(dir_path)
                    audio_files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3'))]
                    print(f"  📊 音频文件数量: {len(audio_files)}")
                    
                    if audio_files:
                        print("  🎵 最近的音频文件:")
                        for f in audio_files[-3:]:  # 显示最后3个文件
                            full_path = os.path.join(dir_path, f)
                            size = os.path.getsize(full_path)
                            print(f"    - {f} ({size} 字节)")
                    
                except Exception as e:
                    print(f"  ⚠️  无法列出文件: {e}")
            else:
                print(f"  ❌ 无写权限")
        else:
            print(f"  ❌ 目录不存在")
    
    return True

def main():
    """主测试函数"""
    print("🔧 音频保存功能调试")
    print("=" * 50)
    
    tests = [
        ("音频保存功能", test_audio_save_function),
        ("音频数据流", test_audio_data_flow),
        ("输出目录检查", check_output_directory),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*15} {test_name} {'='*15}")
            if test_func():
                passed += 1
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 测试通过")
    
    print("\n🔍 可能的问题:")
    print("1. 音频数据传递问题 - 检查生成器输出")
    print("2. 文件权限问题 - 检查输出目录权限")
    print("3. 音频格式问题 - 检查音频数据格式")
    print("4. 路径问题 - 检查输出路径设置")
    
    print("\n🛠️  调试建议:")
    print("1. 检查ComfyUI控制台日志")
    print("2. 验证音频数据不为空")
    print("3. 确认输出目录可写")
    print("4. 尝试不同的文件格式")
    
    print("\n📋 检查清单:")
    print("- [ ] 生成器是否输出了有效音频")
    print("- [ ] 音频时长是否大于0.1秒")
    print("- [ ] 输出目录是否存在且可写")
    print("- [ ] 文件名是否有效")
    print("- [ ] 是否有足够的磁盘空间")

if __name__ == "__main__":
    main()
