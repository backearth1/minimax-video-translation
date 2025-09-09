import requests
import json
import time
import os
from typing import Dict, Any, Optional

class TTSService:
    def __init__(self, config, rate_limiter, logger_service):
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = logger_service
        
    def synthesize_speech(self, text: str, voice_id: str, speed: float = 1.0,
                         output_path: str = None) -> Dict[str, Any]:
        """使用指定音色ID合成语音"""
        # 检查限流
        self.rate_limiter.wait_for_availability('tts')
        
        url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={self.config.group_id}"
        
        # 获取语言增强参数
        language_boost = self._get_language_boost()
        
        payload = {
            "model": self.config.tts_model,
            "text": text,
            "language_boost": language_boost,
            "output_format": "url",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed
            }
        }
        
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                trace_id = response_data.get('trace_id', response.headers.get('Trace-ID', ''))
                
                # 打印详细调试信息
                print(f"TTS API Response Headers: {dict(response.headers)}")
                print(f"TTS API Trace-ID: {trace_id}")
                print(f"TTS API Response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                
                if 'data' in response_data and response_data['data'] and 'audio' in response_data['data']:
                    audio_url = response_data['data']['audio']
                    
                    self.logger.log_api_call(
                        "TTS语音合成", url, trace_id, "success", duration
                    )
                    
                    # 下载音频文件
                    download_result = self.download_audio(audio_url, output_path)
                    
                    if download_result["success"]:
                        return {
                            "success": True,
                            "audio_url": audio_url,
                            "audio_path": download_result["file_path"],
                            "trace_id": trace_id,
                            "text": text,
                            "voice_id": voice_id,
                            "speed": speed
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"音频下载失败: {download_result['error']}",
                            "trace_id": trace_id
                        }
                else:
                    error_msg = "API响应中未找到音频数据"
                    self.logger.log("ERROR", f"TTS合成失败: {error_msg}", trace_id)
                    return {"success": False, "error": error_msg, "trace_id": trace_id}
            else:
                error_msg = f"TTS请求失败: {response.status_code} - {response.text}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"TTS合成异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def download_audio(self, audio_url: str, output_path: str = None) -> Dict[str, Any]:
        """下载音频文件"""
        try:
            if not output_path:
                timestamp = int(time.time())
                output_path = f"./temp/tts_output_{timestamp}.mp3"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            response = requests.get(audio_url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                self.logger.log("INFO", f"音频文件已保存到: {output_path}")
                return {"success": True, "file_path": output_path}
            else:
                return {"success": False, "error": f"下载失败: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": f"下载异常: {str(e)}"}
    
    def _get_language_boost(self) -> str:
        """根据目标语言获取language_boost参数"""
        language_map = {
            "中文": "Chinese",
            "粤语": "Chinese,Yue", 
            "英语": "English",
            "西班牙语": "Spanish",
            "法语": "French",
            "俄语": "Russian",
            "德语": "German",
            "葡萄牙语": "Portuguese",
            "阿拉伯语": "Arabic",
            "意大利语": "Italian",
            "日语": "Japanese",
            "韩语": "Korean"
        }
        
        return language_map.get(self.config.target_language, "Chinese")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except ImportError:
            # 如果没有librosa，使用ffmpeg
            try:
                import subprocess
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_format', audio_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    return float(data['format']['duration'])
                else:
                    return 0.0
            except:
                # 如果都不可用，返回估算值（基于文本长度）
                return len(audio_path) * 0.1  # 粗略估算
        except:
            return 0.0
    
    def trim_silence(self, audio_path: str, output_path: str = None) -> Dict[str, Any]:
        """去除音频开头和结尾的静音"""
        try:
            import librosa
            import soundfile as sf
            
            # 加载音频
            y, sr = librosa.load(audio_path)
            
            # 去除静音
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if not output_path:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_trimmed{ext}"
            
            # 保存处理后的音频
            sf.write(output_path, y_trimmed, sr)
            
            original_duration = len(y) / sr
            trimmed_duration = len(y_trimmed) / sr
            
            return {
                "success": True,
                "output_path": output_path,
                "original_duration": original_duration,
                "trimmed_duration": trimmed_duration,
                "silence_removed": original_duration - trimmed_duration
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "需要安装librosa和soundfile库进行音频处理"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"音频处理失败: {str(e)}"
            }
    
    def batch_synthesize(self, segments: list) -> list:
        """批量合成语音"""
        results = []
        
        for i, segment in enumerate(segments):
            self.logger.log("INFO", f"正在合成第{i+1}/{len(segments)}个语音片段...")
            
            output_path = f"./temp/segment_{segment.get('sequence', i)}_audio.mp3"
            
            result = self.synthesize_speech(
                segment['translated_text'],
                segment['voice_id'],
                segment.get('speed', 1.0),
                output_path
            )
            
            result['segment_id'] = segment.get('sequence', i)
            results.append(result)
            
            # 批量处理时添加延迟
            if i < len(segments) - 1:
                time.sleep(1)
                
        return results