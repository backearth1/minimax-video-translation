#!/usr/bin/env python3
"""
API测试脚本 - 验证MiniMax API集成
"""

import requests
import json
import time
import sys
from pathlib import Path

# 测试配置
GROUP_ID = "1747179187841536150"
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmnZzno4oiLCJVc2VyTmFtZSI6IuadnOejiiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzQ3MTc5MTg3ODQ5OTI0NzU4IiwiUGhvbmUiOiIxMzAyNTQ5MDQyMyIsIkdyb3VwSUQiOiIxNzQ3MTc5MTg3ODQxNTM2MTUwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiZGV2aW5AbWluaW1heGkuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjQtMTItMjMgMTE6NTE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.szVUN2AH7lJ9fQ3EYfzcLcamSCFAOye3Y6yO3Wj_tlNhnhBIYxEEMvZsVgH9mgOe6uhRczOqibmEMbVMUD_1DqtykrbD5klaB4_nhRnDl8fbaAf7m8B1OTRTUIiqgXRVglITenx3K_ugZ6teqiqypByJoLleHbZCSPWvy1-NaDiynb7qAsGzN1V6N4BOTNza1hL5PYdlrXLe2yjQv3YW8nOjQDIGCO1ZqnVBF0UghVaO4V-GZu1Z_0JnkLa7x_2ZXKXAe-LWhk9npwGFzQfLL3aH4oUzlsoEDGnuz3RZdZsFCe95MUiG8dCWfsxhVqlQ5GoFM3LQBAXuLZyqDpmSgg"

def test_llm_translation():
    """测试LLM翻译API"""
    print("🔍 测试LLM翻译API...")
    
    url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "MiniMax-Text-01",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的翻译专家，请将用户提供的文本翻译成英语。"
            },
            {
                "role": "user",
                "content": "请将以下文本翻译成英语：你好，欢迎使用视频翻译测试平台！"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and data['choices']:
                translated = data['choices'][0]['message']['content']
                trace_id = data.get('trace_id', 'N/A')
                print(f"   ✅ 翻译成功!")
                print(f"   📝 结果: {translated}")
                print(f"   🔗 Trace ID: {trace_id}")
                return True
            else:
                print(f"   ❌ 响应格式错误: {data}")
                return False
        else:
            print(f"   ❌ 请求失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False

def test_tts_api():
    """测试TTS API"""
    print("\n🔍 测试TTS语音合成API...")
    
    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={GROUP_ID}"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "speech-01-turbo",
        "text": "Hello, welcome to the video translation platform!",
        "language_boost": "English",
        "output_format": "url",
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1.0
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data'] and 'audio' in data['data']:
                audio_url = data['data']['audio']
                trace_id = data.get('trace_id', 'N/A')
                print(f"   ✅ TTS合成成功!")
                print(f"   🎵 音频URL: {audio_url}")
                print(f"   🔗 Trace ID: {trace_id}")
                return True
            else:
                print(f"   ❌ 响应中未找到音频数据: {data}")
                return False
        else:
            print(f"   ❌ 请求失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False

def test_voice_clone_upload():
    """测试音色克隆上传API（模拟测试）"""
    print("\n🔍 测试音色克隆上传API...")
    
    # 由于需要实际的音频文件，这里只测试API端点的可达性
    url = f'https://api.minimax.chat/v1/files/upload?GroupId={GROUP_ID}'
    headers = {
        'authority': 'api.minimax.chat',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    try:
        # 只发送一个HEAD请求测试端点
        response = requests.head(url, headers=headers, timeout=10)
        if response.status_code in [200, 400, 401]:  # 400表示缺少文件，这是预期的
            print(f"   ✅ 音色克隆上传API端点可达")
            print(f"   📡 状态码: {response.status_code}")
            return True
        else:
            print(f"   ❌ 端点异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 连接异常: {e}")
        return False

def test_dependencies():
    """测试项目依赖"""
    print("\n🔍 测试项目依赖...")
    
    dependencies = [
        ('requests', 'requests'),
        ('flask', 'Flask'),
        ('json', 'json (内置)'),
        ('pathlib', 'pathlib (内置)'),
        ('datetime', 'datetime (内置)')
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - 未安装")
            missing.append(name)
    
    optional_deps = [
        ('librosa', 'librosa (音频处理)'),
        ('cv2', 'opencv-python (视频处理)'),
        ('numpy', 'numpy (数值计算)')
    ]
    
    print("\n   可选依赖:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - 未安装 (可选)")
    
    return len(missing) == 0

def main():
    """主测试函数"""
    print("🚀 MiniMax API集成测试")
    print("=" * 50)
    
    # 基本信息
    print(f"📋 测试配置:")
    print(f"   Group ID: {GROUP_ID}")
    print(f"   API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
    print("")
    
    # 依赖测试
    deps_ok = test_dependencies()
    
    # API测试
    llm_ok = test_llm_translation()
    tts_ok = test_tts_api()
    clone_ok = test_voice_clone_upload()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   依赖检查: {'✅ 通过' if deps_ok else '❌ 失败'}")
    print(f"   LLM翻译: {'✅ 通过' if llm_ok else '❌ 失败'}")
    print(f"   TTS合成: {'✅ 通过' if tts_ok else '❌ 失败'}")
    print(f"   音色克隆: {'✅ 通过' if clone_ok else '❌ 失败'}")
    
    all_ok = deps_ok and llm_ok and tts_ok and clone_ok
    
    if all_ok:
        print("\n🎉 所有测试通过！可以开始使用视频翻译平台。")
        print("\n📝 下一步:")
        print("   1. 运行: python3 run.py")
        print("   2. 访问: https://localhost:5555")
        print("   3. 上传测试视频开始翻译")
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接。")
        if not deps_ok:
            print("   - 请安装缺失的依赖包")
        if not (llm_ok and tts_ok):
            print("   - 请检查API Key和Group ID配置")
        if not clone_ok:
            print("   - 请检查网络连接到MiniMax服务")
    
    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)