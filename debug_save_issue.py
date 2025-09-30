#!/usr/bin/env python3
"""
调试音频保存问题

专门调试为什么SongBloomAudioSaver显示0.093秒的问题
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def analyze_short_audio_issue():
    """分析短音频问题"""
    print("🔍 分析0.093秒音频问题...")
    
    # 0.093秒在48kHz采样率下的样本数
    sample_rate = 48000
    duration = 0.093
    expected_samples = int(sample_rate * duration)
    
    print(f"📊 0.093秒音频分析:")
    print(f"  - 采样率: {sample_rate}Hz")
    print(f"  - 时长: {duration}秒")
    print(f"  - 预期样本数: {expected_samples}")
    print(f"  - 预期文件大小: ~{expected_samples * 4}字节 (float32)")
    
    # 检查可能的原因
    possible_causes = [
        "生成器输出的音频数据太短",
        "音频数据在传递过程中被截断",
        "音频数据格式不正确",
        "生成过程中出现错误，返回了默认的短音频",
        "音频数据的维度处理有问题"
    ]
    
    print(f"\n🤔 可能的原因:")
    for i, cause in enumerate(possible_causes, 1):
        print(f"  {i}. {cause}")
    
    return True

def create_problematic_audio_scenarios():
    """创建可能有问题的音频场景"""
    print("\n🧪 创建问题音频场景...")
    
    scenarios = []
    
    # 场景1: 极短音频 (0.093秒)
    sample_rate = 48000
    short_samples = int(48000 * 0.093)  # 4464个样本
    short_audio = {
        "waveform": torch.randn(1, short_samples),
        "sample_rate": sample_rate
    }
    scenarios.append(("极短音频(0.093s)", short_audio))
    
    # 场景2: 错误的维度
    wrong_dim_audio = {
        "waveform": torch.randn(1, 1, 4464),  # 3维
        "sample_rate": sample_rate
    }
    scenarios.append(("错误维度", wrong_dim_audio))
    
    # 场景3: 空音频
    empty_audio = {
        "waveform": torch.zeros(1, 0),  # 空
        "sample_rate": sample_rate
    }
    scenarios.append(("空音频", empty_audio))
    
    # 场景4: 单样本音频
    single_sample_audio = {
        "waveform": torch.randn(1, 1),  # 只有1个样本
        "sample_rate": sample_rate
    }
    scenarios.append(("单样本", single_sample_audio))
    
    # 场景5: 正常但很短的音频
    normal_short_audio = {
        "waveform": torch.randn(1, 1000),  # 1000个样本 ≈ 0.021秒
        "sample_rate": sample_rate
    }
    scenarios.append(("正常短音频", normal_short_audio))
    
    for name, audio in scenarios:
        waveform = audio["waveform"]
        duration = waveform.shape[-1] / audio["sample_rate"]
        print(f"  📊 {name}: {waveform.shape}, {duration:.3f}秒")
    
    return scenarios

def simulate_generator_output():
    """模拟生成器可能的输出"""
    print("\n🎭 模拟生成器输出...")
    
    # 模拟不同的生成器输出情况
    outputs = []
    
    # 正常输出
    normal_output = {
        "waveform": torch.randn(1, 48000 * 5),  # 5秒
        "sample_rate": 48000
    }
    outputs.append(("正常输出", normal_output))
    
    # 错误输出 - 返回了错误处理的默认音频
    error_output = {
        "waveform": torch.zeros(1, 1, 48000),  # 1秒静音，3维
        "sample_rate": 48000
    }
    outputs.append(("错误处理输出", error_output))
    
    # 生成失败 - 返回极短音频
    failed_output = {
        "waveform": torch.zeros(1, 4464),  # 0.093秒
        "sample_rate": 48000
    }
    outputs.append(("生成失败输出", failed_output))
    
    # 维度错误
    dim_error_output = {
        "waveform": torch.randn(1, 1, 1000),  # 错误的3维
        "sample_rate": 48000
    }
    outputs.append(("维度错误输出", dim_error_output))
    
    for name, output in outputs:
        waveform = output["waveform"]
        duration = waveform.shape[-1] / output["sample_rate"]
        print(f"  📊 {name}: {waveform.shape}, {duration:.3f}秒")
        
        # 检查这个输出传递给保存器会发生什么
        print(f"    传递给保存器后:")
        save_waveform = waveform.clone()
        if save_waveform.ndim == 3:
            save_waveform = save_waveform.squeeze(0)
            print(f"    - squeeze后: {save_waveform.shape}")
        
        final_duration = save_waveform.shape[-1] / output["sample_rate"]
        print(f"    - 最终时长: {final_duration:.3f}秒")
        
        if abs(final_duration - 0.093) < 0.001:
            print(f"    ⚠️  这个输出会导致0.093秒的问题！")
        print()
    
    return outputs

def check_comfyui_logs():
    """检查ComfyUI日志中的线索"""
    print("\n📋 检查日志线索...")
    
    # 提供检查日志的指导
    log_checks = [
        "查看ComfyUI控制台是否有SongBloom相关错误",
        "检查是否有'Error generating song'消息",
        "查看是否有音频形状相关的警告",
        "检查是否有文件保存相关的错误",
        "查看是否有内存不足的警告"
    ]
    
    print("🔍 需要检查的日志内容:")
    for i, check in enumerate(log_checks, 1):
        print(f"  {i}. {check}")
    
    print("\n💡 关键日志关键词:")
    keywords = [
        "SongBloom",
        "Error generating",
        "Audio saved",
        "waveform shape",
        "duration",
        "0.093",
        "empty audio",
        "torch.zeros"
    ]
    
    for keyword in keywords:
        print(f"  - '{keyword}'")
    
    return True

def provide_debugging_steps():
    """提供调试步骤"""
    print("\n🛠️  调试步骤建议:")
    
    steps = [
        "重启ComfyUI以应用增强的调试日志",
        "运行一个简单的生成任务",
        "查看控制台中的详细调试信息",
        "检查'=== SongBloomAudioSaver Debug Info ==='部分",
        "确认输入音频的形状和时长",
        "检查生成器是否真的生成了长音频",
        "验证音频数据在传递过程中是否被修改"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    
    print("\n🎯 重点检查:")
    important_checks = [
        "生成器的'Generated song successfully. Duration: XXXs'消息",
        "保存器的'Input audio shape'和'Audio duration'信息",
        "是否有'Audio is very short'警告",
        "最终保存的文件大小是否合理"
    ]
    
    for check in important_checks:
        print(f"  ⭐ {check}")
    
    return True

def main():
    """主函数"""
    print("🔧 调试音频保存问题 - 0.093秒问题分析")
    print("=" * 60)
    
    tests = [
        ("分析短音频问题", analyze_short_audio_issue),
        ("创建问题音频场景", create_problematic_audio_scenarios),
        ("模拟生成器输出", simulate_generator_output),
        ("检查日志线索", check_comfyui_logs),
        ("提供调试步骤", provide_debugging_steps),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*15} {test_name} {'='*15}")
            test_func()
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 总结:")
    print("1. 0.093秒 ≈ 4464个样本 (48kHz)")
    print("2. 这可能是生成器返回的错误处理音频")
    print("3. 需要检查生成器的实际输出")
    print("4. 增强的调试日志会提供更多信息")
    
    print("\n🚀 下一步:")
    print("1. 重启ComfyUI")
    print("2. 运行生成任务")
    print("3. 查看详细的调试日志")
    print("4. 根据日志信息定位问题")
    
    print("\n📞 如果问题持续:")
    print("- 提供完整的控制台日志")
    print("- 说明使用的具体工作流")
    print("- 描述生成器显示的时长信息")

if __name__ == "__main__":
    main()
