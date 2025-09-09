#!/usr/bin/env python3
"""
快速重启脚本 - 重新加载代码并启动
"""

import os
import sys
import subprocess
import signal
import time

def find_and_kill_existing():
    """查找并终止现有的Flask进程"""
    try:
        # 查找运行在5555端口的进程
        result = subprocess.run(['lsof', '-ti:5555'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"✅ 终止进程 PID: {pid}")
                except:
                    pass
            time.sleep(2)
    except:
        pass

def start_server():
    """启动服务器"""
    print("🚀 启动视频翻译平台 (真实API版本)...")
    print("=" * 50)
    print("📋 真实功能:")
    print("   ✅ ffmpeg音频提取")
    print("   ✅ Whisper ASR语音识别")
    print("   ✅ MiniMax LLM翻译")
    print("   ✅ MiniMax音色克隆")
    print("   ✅ MiniMax TTS合成")
    print("   ✅ 5步时间戳对齐算法")
    print("=" * 50)
    
    # 启动Flask应用
    os.execv(sys.executable, [sys.executable, 'run.py'])

if __name__ == '__main__':
    find_and_kill_existing()
    start_server()