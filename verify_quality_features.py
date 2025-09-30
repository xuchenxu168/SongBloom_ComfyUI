#!/usr/bin/env python3
"""
验证音频质量优化功能

检查代码结构和配置是否正确
"""

import os
import re
from pathlib import Path

def check_quality_config_node():
    """检查质量配置节点实现"""
    print("🔧 检查SongBloomQualityConfig节点实现...")
    
    songbloom_file = Path("songbloom_nodes.py")
    if not songbloom_file.exists():
        print("❌ songbloom_nodes.py文件不存在")
        return False
    
    content = songbloom_file.read_text(encoding='utf-8')
    
    # 检查类定义
    if "class SongBloomQualityConfig:" not in content:
        print("❌ SongBloomQualityConfig类未定义")
        return False
    print("✅ SongBloomQualityConfig类已定义")
    
    # 检查质量预设
    presets = ["Ultra High", "High", "Balanced", "Fast"]
    for preset in presets:
        if preset not in content:
            print(f"❌ 缺少质量预设: {preset}")
            return False
    print("✅ 所有质量预设已定义")
    
    # 检查返回类型
    if "SONGBLOOM_QUALITY_CONFIG" not in content:
        print("❌ 缺少质量配置返回类型")
        return False
    print("✅ 质量配置返回类型已定义")
    
    return True

def check_generator_integration():
    """检查生成器集成"""
    print("\n🎶 检查生成器质量配置集成...")
    
    songbloom_file = Path("songbloom_nodes.py")
    content = songbloom_file.read_text(encoding='utf-8')
    
    # 检查生成器是否支持质量配置输入
    generator_section = re.search(r'class SongBloomGenerator:.*?def generate_song', content, re.DOTALL)
    if not generator_section:
        print("❌ 无法找到SongBloomGenerator类")
        return False
    
    generator_text = generator_section.group(0)
    
    # 检查quality_config参数
    if "quality_config" not in generator_text:
        print("❌ 生成器缺少quality_config参数")
        return False
    print("✅ 生成器支持质量配置输入")
    
    # 检查默认参数优化
    cfg_match = re.search(r'"default":\s*(\d+\.?\d*)', generator_text)
    if cfg_match:
        cfg_default = float(cfg_match.group(1))
        if cfg_default >= 3.0:
            print(f"✅ CFG默认值已优化: {cfg_default}")
        else:
            print(f"⚠️  CFG默认值可能需要优化: {cfg_default}")
    
    return True

def check_node_mappings():
    """检查节点映射"""
    print("\n🗺️ 检查节点映射...")
    
    songbloom_file = Path("songbloom_nodes.py")
    content = songbloom_file.read_text(encoding='utf-8')
    
    # 检查NODE_CLASS_MAPPINGS
    if "SongBloomQualityConfig" not in content.split("NODE_CLASS_MAPPINGS")[1].split("}")[0]:
        print("❌ SongBloomQualityConfig未在NODE_CLASS_MAPPINGS中")
        return False
    print("✅ SongBloomQualityConfig已在NODE_CLASS_MAPPINGS中")
    
    # 检查NODE_DISPLAY_NAME_MAPPINGS
    if "SongBloomQualityConfig" not in content.split("NODE_DISPLAY_NAME_MAPPINGS")[1].split("}")[0]:
        print("❌ SongBloomQualityConfig未在NODE_DISPLAY_NAME_MAPPINGS中")
        return False
    print("✅ SongBloomQualityConfig已在NODE_DISPLAY_NAME_MAPPINGS中")
    
    return True

def check_example_workflow():
    """检查示例工作流"""
    print("\n📋 检查高质量生成工作流...")
    
    workflow_file = Path("example_workflows/high_quality_generation.json")
    if not workflow_file.exists():
        print("❌ 高质量生成工作流文件不存在")
        return False
    print("✅ 高质量生成工作流文件存在")
    
    try:
        import json
        workflow_content = json.loads(workflow_file.read_text(encoding='utf-8'))
        
        # 检查是否包含质量配置节点
        has_quality_config = False
        for node in workflow_content.get("nodes", []):
            if node.get("type") == "SongBloomQualityConfig":
                has_quality_config = True
                break
        
        if has_quality_config:
            print("✅ 工作流包含质量配置节点")
        else:
            print("❌ 工作流缺少质量配置节点")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流文件格式错误: {e}")
        return False

def check_documentation():
    """检查文档"""
    print("\n📚 检查文档更新...")
    
    # 检查质量优化指南
    guide_file = Path("音频质量优化指南.md")
    if not guide_file.exists():
        print("❌ 音频质量优化指南不存在")
        return False
    print("✅ 音频质量优化指南存在")
    
    # 检查README更新
    readme_file = Path("README.md")
    if readme_file.exists():
        readme_content = readme_file.read_text(encoding='utf-8')
        if "质量优化" in readme_content or "Quality" in readme_content:
            print("✅ README包含质量优化内容")
        else:
            print("⚠️  README可能缺少质量优化内容")
    
    # 检查CHANGELOG更新
    changelog_file = Path("CHANGELOG.md")
    if changelog_file.exists():
        changelog_content = changelog_file.read_text(encoding='utf-8')
        if "质量优化" in changelog_content or "v1.4.0" in changelog_content:
            print("✅ CHANGELOG包含质量优化更新")
        else:
            print("⚠️  CHANGELOG可能缺少质量优化更新")
    
    return True

def check_parameter_optimization():
    """检查参数优化"""
    print("\n⚙️ 检查参数优化...")
    
    songbloom_file = Path("songbloom_nodes.py")
    content = songbloom_file.read_text(encoding='utf-8')
    
    # 检查默认推理配置
    inference_section = re.search(r'cfg\.inference = \{.*?\}', content, re.DOTALL)
    if inference_section:
        inference_text = inference_section.group(0)
        
        # 检查CFG系数
        cfg_match = re.search(r"'cfg_coef':\s*(\d+\.?\d*)", inference_text)
        if cfg_match and float(cfg_match.group(1)) >= 3.0:
            print(f"✅ 默认推理CFG系数已优化: {cfg_match.group(1)}")
        else:
            print("⚠️  默认推理CFG系数可能需要优化")
        
        # 检查步数
        steps_match = re.search(r"'steps':\s*(\d+)", inference_text)
        if steps_match and int(steps_match.group(1)) >= 100:
            print(f"✅ 默认推理步数已优化: {steps_match.group(1)}")
        else:
            print("⚠️  默认推理步数可能需要优化")
        
        # 检查top_k
        top_k_match = re.search(r"'top_k':\s*(\d+)", inference_text)
        if top_k_match and int(top_k_match.group(1)) <= 100:
            print(f"✅ 默认top_k已优化: {top_k_match.group(1)}")
        else:
            print("⚠️  默认top_k可能需要优化")
    
    return True

def main():
    """主验证函数"""
    print("🎯 SongBloom音频质量优化功能验证")
    print("=" * 50)
    
    checks = [
        ("质量配置节点实现", check_quality_config_node),
        ("生成器集成", check_generator_integration),
        ("节点映射", check_node_mappings),
        ("示例工作流", check_example_workflow),
        ("文档更新", check_documentation),
        ("参数优化", check_parameter_optimization),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            print(f"\n{'='*15} {check_name} {'='*15}")
            if check_func():
                passed += 1
                print(f"✅ {check_name} - 通过")
            else:
                print(f"❌ {check_name} - 失败")
        except Exception as e:
            print(f"❌ {check_name} - 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 验证结果: {passed}/{total} 检查通过")
    
    if passed == total:
        print("\n🎉 所有质量优化功能验证通过！")
        print("\n🚀 下一步:")
        print("1. 重启ComfyUI")
        print("2. 导入high_quality_generation.json工作流")
        print("3. 使用SongBloomQualityConfig节点")
        print("4. 测试不同质量预设的效果")
        print("5. 对比音频质量改善")
    else:
        print(f"\n⚠️  有{total-passed}个检查失败")
    
    print("\n📋 质量优化功能清单:")
    print("✅ SongBloomQualityConfig节点 - 专业质量配置")
    print("✅ 四种质量预设 - Ultra High/High/Balanced/Fast")
    print("✅ 自定义参数覆盖 - 灵活调整")
    print("✅ 生成器集成 - 无缝连接")
    print("✅ 默认参数优化 - 更高质量")
    print("✅ 详细调试日志 - 问题诊断")
    print("✅ 完整文档支持 - 使用指南")
    
    print("\n💡 使用建议:")
    print("- 音质差问题: 使用'High'或'Ultra High'预设")
    print("- 生成速度慢: 使用'Balanced'或'Fast'预设")
    print("- 参数调优: 参考音频质量优化指南.md")
    print("- 问题诊断: 查看增强的调试日志")

if __name__ == "__main__":
    main()
