#!/usr/bin/env python3
"""
TTS API调试脚本
"""

import requests
import json

# 测试配置
GROUP_ID = "1747179187841536150"
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmnZzno4oiLCJVc2VyTmFtZSI6IuadnOejiiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzQ3MTc5MTg3ODQ5OTI0NzU4IiwiUGhvbmUiOiIxMzAyNTQ5MDQyMyIsIkdyb3VwSUQiOiIxNzQ3MTc5MTg3ODQxNTM2MTUwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiZGV2aW5AbWluaW1heGkuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjQtMTItMjMgMTE6NTE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.szVUN2AH7lJ9fQ3EYfzcLcamSCFAOye3Y6yO3Wj_tlNhnhBIYxEEMvZsVgH9mgOe6uhRczOqibmEMbVMUD_1DqtykrbD5klaB4_nhRnDl8fbaAf7m8B1OTRTUIiqgXRVglITenx3K_ugZ6teqiqypByJoLleHbZCSPWvy1-NaDiynb7qAsGzN1V6N4BOTNza1hL5PYdlrXLe2yjQv3YW8nOjQDIGCO1ZqnVBF0UghVaO4V-GZu1Z_0JnkLa7x_2ZXKXAe-LWhk9npwGFzQfLL3aH4oUzlsoEDGnuz3RZdZsFCe95MUiG8dCWfsxhVqlQ5GoFM3LQBAXuLZyqDpmSgg"

def test_tts_api():
    """测试TTS API"""
    print("🔍 测试TTS API...")
    
    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={GROUP_ID}"
    
    payload = {
        "model": "speech-2.5-hd-preview",
        "text": "Hello everyone, welcome to the video translation platform",
        "language_boost": "English",
        "output_format": "url",
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1.0
        }
    }
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"📤 请求URL: {url}")
        print(f"📤 请求头: {headers}")
        print(f"📤 请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        print("-" * 50)
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"📥 响应状态码: {response.status_code}")
        print(f"📥 响应头: {dict(response.headers)}")
        
        if 'Trace-ID' in response.headers:
            print(f"🔗 Trace-ID: {response.headers['Trace-ID']}")
        
        try:
            response_data = response.json()
            print(f"📥 响应体: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            # 检查响应结构
            if 'data' in response_data:
                print("✅ 找到 'data' 字段")
                if response_data['data'] and 'audio' in response_data['data']:
                    print("✅ 找到 'audio' 字段")
                    audio_url = response_data['data']['audio']
                    print(f"🎵 音频URL: {audio_url}")
                else:
                    print("❌ 'data' 字段为空或缺少 'audio'")
            else:
                print("❌ 响应中缺少 'data' 字段")
                
        except json.JSONDecodeError:
            print(f"❌ 响应不是有效的JSON: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")

if __name__ == '__main__':
    test_tts_api()