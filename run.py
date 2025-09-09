#!/usr/bin/env python3
"""
视频翻译测试平台启动脚本
运行方式: python run.py
访问地址: https://localhost:5555
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_translator.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    """创建必要的目录"""
    directories = [
        'temp',
        'logs', 
        'uploads',
        'output'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        
def check_dependencies():
    """检查依赖包"""
    # 包名映射：安装名 -> 导入名
    package_mapping = {
        'flask': 'flask',
        'requests': 'requests', 
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'opencv-python': 'cv2'
    }
    
    missing_packages = []
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n或者安装完整依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 启动视频翻译测试平台...")
    print("=" * 50)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建目录
    create_directories()
    logger.info("创建必要目录完成")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    logger.info("依赖检查通过")
    
    # 导入Flask应用
    try:
        from app import app
    except ImportError as e:
        logger.error(f"导入应用失败: {e}")
        print("❌ 应用导入失败，请检查代码完整性")
        sys.exit(1)
    
    # 配置信息
    print("\n📋 配置信息:")
    print(f"   项目目录: {project_root}")
    print(f"   访问地址: https://localhost:5555")
    print(f"   调试模式: {'开启' if app.debug else '关闭'}")
    print(f"   日志文件: {project_root}/video_translator.log")
    
    # API配置提醒
    print("\n⚠️  使用提醒:")
    print("   1. 请确保已正确配置Group ID和API Key")
    print("   2. 支持的视频格式: mp4, avi, mov, mkv等")
    print("   3. 视频文件建议不超过5分钟")
    print("   4. 首次使用建议先测试短视频")
    
    print("\n🎯 核心功能:")
    print("   ✅ 视频上传与音频提取")
    print("   ✅ ASR语音识别与智能切分")  
    print("   ✅ LLM逐句翻译")
    print("   ✅ 逐句音色克隆")
    print("   ✅ TTS语音合成")
    print("   ✅ 5步智能时间戳对齐")
    print("   ✅ SRT字幕导入导出")
    print("   ✅ 可视化编辑界面")
    
    print("\n" + "=" * 50)
    logger.info("开始启动Flask应用...")
    
    try:
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5555,
            debug=True,
            ssl_context='adhoc',  # 自动生成SSL证书
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")
        print("\n👋 感谢使用视频翻译测试平台!")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()