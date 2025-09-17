import requests
import json
import time
import os
from typing import Dict, Any, Optional

class VoiceCloneService:
    def __init__(self, config, rate_limiter, logger_service):
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = logger_service
        
    def upload_audio_file(self, audio_file_path: str) -> Dict[str, Any]:
        """上传音频文件用于音色克隆"""
        if not os.path.exists(audio_file_path):
            return {"success": False, "error": "音频文件不存在"}
        
        # 检查限流
        self.rate_limiter.wait_for_availability('clone')
        
        url = f'https://api.minimax.chat/v1/files/upload?GroupId={self.config.group_id}'
        headers = {
            'authority': 'api.minimax.chat',
            'Authorization': f'Bearer {self.config.api_key}'
        }
        
        data = {'purpose': 'voice_clone'}
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                
                start_time = time.time()
                response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    response_data = response.json()
                    trace_id = response_data.get('trace_id', response.headers.get('Trace-ID', ''))
                    
                    # 打印详细调试信息
                    print(f"File Upload API Response Headers: {dict(response.headers)}")
                    print(f"File Upload API Trace-ID: {trace_id}")
                    print(f"File Upload API Response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                    
                    if 'file' in response_data and 'file_id' in response_data['file']:
                        file_id = response_data['file']['file_id']
                        
                        self.logger.log_api_call(
                            "音频文件上传", url, trace_id, "success", duration
                        )
                        
                        return {
                            "success": True,
                            "file_id": file_id,
                            "trace_id": trace_id,
                            "file_path": audio_file_path
                        }
                    else:
                        error_msg = "文件上传响应格式错误"
                        self.logger.log("ERROR", f"文件上传失败: {error_msg}", trace_id)
                        return {"success": False, "error": error_msg}
                else:
                    error_msg = f"文件上传失败: {response.status_code} - {response.text}"
                    self.logger.log("ERROR", error_msg)
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            error_msg = f"文件上传异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def clone_voice(self, file_id: str, voice_id: str) -> Dict[str, Any]:
        """执行音色克隆"""
        # 检查限流
        self.rate_limiter.wait_for_availability('clone')
        
        url = f"https://api.minimax.chat/v1/voice_clone?GroupId={self.config.group_id}"
        
        payload = {
            "file_id": file_id,
            "voice_id": voice_id,
            "need_volumn_normalization": True
        }
        
        headers = {
            'authorization': f'Bearer {self.config.api_key}',
            'content-type': 'application/json'
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                trace_id = response_data.get('trace_id', response.headers.get('Trace-ID', ''))
                
                # 打印详细调试信息
                print(f"Voice Clone API Response Headers: {dict(response.headers)}")
                print(f"Voice Clone API Trace-ID: {trace_id}")
                print(f"Voice Clone API Response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                
                self.logger.log_api_call(
                    "音色克隆", url, trace_id, "success", duration
                )
                
                # 检查新的响应格式
                if 'base_resp' in response_data and response_data['base_resp']['status_code'] == 0:
                    return {
                        "success": True,
                        "response_data": response_data,
                        "trace_id": trace_id,
                        "voice_id": voice_id,
                        "demo_audio": response_data.get('demo_audio', ''),
                        "status_msg": response_data['base_resp']['status_msg']
                    }
                else:
                    error_msg = f"音色克隆失败: {response_data.get('base_resp', {}).get('status_msg', '未知错误')}"
                    self.logger.log("ERROR", error_msg)
                    return {"success": False, "error": error_msg}
            else:
                error_msg = f"音色克隆失败: {response.status_code} - {response.text}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"音色克隆异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def clone_voice_from_audio(self, audio_file_path: str, voice_id: str, 
                              text: str = None, language_boost: str = "Chinese") -> Dict[str, Any]:
        """从音频文件直接进行音色克隆（两步合一）"""
        
        # 步骤1: 检查音频长度，如果小于10秒则复制多次
        processed_audio_path = self._ensure_minimum_duration(audio_file_path, min_duration=10.0)
        
        # 步骤2: 上传音频文件
        self.logger.log("INFO", f"上传音频文件进行音色克隆: {voice_id}")
        upload_result = self.upload_audio_file(processed_audio_path)
        
        if not upload_result["success"]:
            return upload_result
        
        file_id = upload_result["file_id"]
        
        # 步骤3: 执行音色克隆
        self.logger.log("INFO", f"执行音色克隆: {voice_id}")
        clone_result = self.clone_voice(file_id, voice_id)
        
        if clone_result["success"]:
            # 合并结果
            clone_result["upload_trace_id"] = upload_result["trace_id"]
            clone_result["file_id"] = file_id
            clone_result["processed_audio_path"] = processed_audio_path  # 保存处理后的音频路径
            
        return clone_result
    
    def _ensure_minimum_duration(self, audio_file_path: str, min_duration: float = 10.0) -> str:
        """确保音频文件满足最小时长要求，不足则复制多次"""
        try:
            # 获取音频时长
            import librosa
            y, sr = librosa.load(audio_file_path)
            current_duration = len(y) / sr
            
            self.logger.log("INFO", f"原始音频时长: {current_duration:.2f}秒")
            
            if current_duration >= min_duration:
                return audio_file_path
            
            # 计算需要复制的次数
            repeat_times = int(min_duration / current_duration) + 1
            self.logger.log("INFO", f"音频时长不足{min_duration}秒，将复制{repeat_times}次")
            
            # 创建复制后的音频
            import numpy as np
            import soundfile as sf
            
            repeated_audio = np.tile(y, repeat_times)
            
            # 保存到新文件
            import os
            base, ext = os.path.splitext(audio_file_path)
            output_path = f"{base}_extended{ext}"
            sf.write(output_path, repeated_audio, sr)
            
            final_duration = len(repeated_audio) / sr
            self.logger.log("INFO", f"扩展后音频时长: {final_duration:.2f}秒，保存到: {output_path}")
            
            return output_path
            
        except ImportError:
            self.logger.log("WARNING", "缺少librosa或soundfile库，无法检查音频时长，使用原始文件")
            return audio_file_path
        except Exception as e:
            self.logger.log("WARNING", f"音频时长检查失败: {str(e)}，使用原始文件")
            return audio_file_path
    
    def generate_voice_id(self, segment_id: int) -> str:
        """生成唯一的voice_id"""
        timestamp = int(time.time())
        return f"voice_{segment_id}_{timestamp}"
    
    def generate_voice_id_for_speaker(self, speaker_id: str, sequence: int) -> str:
        """为特定说话人生成音色ID"""
        timestamp = int(time.time())
        # 使用说话人ID作为前缀，确保同一说话人使用一致的音色
        return f"voice_{speaker_id}_seq{sequence}_{timestamp}"
    
    def get_language_boost_from_target_language(self, target_language: str) -> str:
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
            "韩语": "Korean",
            "印尼语": "Indonesian",
            "越南语": "Vietnamese",
            "土耳其语": "Turkish",
            "荷兰语": "Dutch",
            "乌克兰语": "Ukrainian",
            "泰语": "Thai",
            "波兰语": "Polish"
        }
        
        return language_map.get(target_language, "Chinese")
    
    def batch_clone_voices(self, audio_segments: list) -> list:
        """批量克隆音色"""
        results = []
        
        for i, segment in enumerate(audio_segments):
            self.logger.log("INFO", f"正在克隆第{i+1}/{len(audio_segments)}个音频片段...")
            
            voice_id = self.generate_voice_id(segment.get('sequence', i))
            language_boost = self.get_language_boost_from_target_language(self.config.target_language)
            
            result = self.clone_voice_from_audio(
                segment['audio_path'],
                voice_id,
                segment['text'],
                language_boost
            )
            
            result['segment_id'] = segment.get('sequence', i)
            results.append(result)
            
            # 批量处理时添加延迟
            if i < len(audio_segments) - 1:
                time.sleep(1)
                
        return results