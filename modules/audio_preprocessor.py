import os
import subprocess
import time
from typing import Dict, Any, Optional
from pathlib import Path

class AudioPreprocessor:
    """音频预处理服务，专门负责人声提取、背景音分离等处理"""
    
    def __init__(self, logger_service):
        self.logger = logger_service
    
    def extract_voice(self, input_path: str, voice_output_path: str = None,
                     background_output_path: str = None) -> Dict[str, Any]:
        """
        从音频中分离人声和背景音
        
        Args:
            input_path: 输入音频文件路径
            voice_output_path: 人声输出路径，如果为None则自动生成
            background_output_path: 背景音输出路径，如果为None则自动生成
            
        Returns:
            处理结果字典，包含人声和背景音路径
        """
        try:
            input_name = Path(input_path).stem
            
            if not voice_output_path:
                voice_output_path = f"./temp/{input_name}_voice.wav"
            if not background_output_path:
                background_output_path = f"./temp/{input_name}_background.wav"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(voice_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(background_output_path), exist_ok=True)
            
            self.logger.log("INFO", "开始人声和背景音分离...")
            
            # 方法1：使用中央声道提取（适用于大部分音乐和视频）
            voice_result = self._extract_center_channel(input_path, voice_output_path)
            
            # 方法2：提取背景音（原音频减去人声部分）  
            background_result = self._extract_background(input_path, voice_output_path, background_output_path)
            
            if voice_result["success"] and background_result["success"]:
                # 获取音频信息
                voice_duration = self._get_audio_duration(voice_output_path)
                voice_size = os.path.getsize(voice_output_path)
                background_duration = self._get_audio_duration(background_output_path)
                background_size = os.path.getsize(background_output_path)
                
                self.logger.log("INFO", f"人声提取完成: 时长{voice_duration:.2f}s, 大小{voice_size/1024/1024:.2f}MB")
                self.logger.log("INFO", f"背景音提取完成: 时长{background_duration:.2f}s, 大小{background_size/1024/1024:.2f}MB")
                
                return {
                    "success": True,
                    "voice_path": voice_output_path,
                    "background_path": background_output_path,
                    "voice_duration": voice_duration,
                    "background_duration": background_duration,
                    "voice_size": voice_size,
                    "background_size": background_size
                }
            else:
                error_msgs = []
                if not voice_result["success"]:
                    error_msgs.append(f"人声提取失败: {voice_result['error']}")
                if not background_result["success"]:
                    error_msgs.append(f"背景音提取失败: {background_result['error']}")
                
                return {"success": False, "error": "; ".join(error_msgs)}
            
        except Exception as e:
            error_msg = f"音频预处理异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def _extract_center_channel(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """提取中央声道（人声）"""
        try:
            # 使用中央声道提取技术分离人声
            cmd = [
                'ffmpeg', '-i', input_path,
                '-af', 'pan=mono|c0=0.5*c0+0.5*c1,highpass=f=80,lowpass=f=8000,dynaudnorm=f=500:g=3:r=0.3',
                # pan=mono: 将立体声混合为单声道，保留中央部分（通常是人声）
                # highpass=f=80: 去除80Hz以下的低频噪音
                # lowpass=f=8000: 去除8000Hz以上的高频噪音
                # dynaudnorm: 动态标准化，增强人声
                '-ar', '16000',  # 采样率16kHz，适合语音识别
                '-ac', '1',      # 单声道
                '-y', output_path
            ]
            
            self.logger.log("INFO", "开始提取人声...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": f"人声提取失败: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "人声提取超时"}
        except Exception as e:
            return {"success": False, "error": f"人声提取异常: {str(e)}"}
    
    def _extract_background(self, original_path: str, voice_path: str, output_path: str) -> Dict[str, Any]:
        """提取背景音（原音频减去人声）"""
        try:
            # 使用karaoke滤镜提取背景音乐
            cmd = [
                'ffmpeg', '-i', original_path,
                '-af', 'pan=mono|c0=0.5*c0+-0.5*c1',
                # 使用相位抵消技术去除中央声道（人声），保留左右声道差异（通常是背景音乐）
                '-y', output_path
            ]
            
            self.logger.log("INFO", "开始提取背景音...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {"success": True, "output_path": output_path}
            else:
                return {"success": False, "error": f"背景音提取失败: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "背景音提取超时"}
        except Exception as e:
            return {"success": False, "error": f"背景音提取异常: {str(e)}"}
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _cleanup_temp_files(self, file_paths: list, keep: str = None):
        """清理临时文件"""
        for file_path in file_paths:
            if file_path and file_path != keep and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.log("DEBUG", f"清理临时文件: {file_path}")
                except Exception as e:
                    self.logger.log("WARNING", f"清理临时文件失败: {file_path} - {str(e)}")
    
    def analyze_audio_content(self, audio_path: str) -> Dict[str, Any]:
        """分析音频内容，判断是否需要人声背景音分离"""
        try:
            # 使用ffprobe分析音频特征
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # 获取音频流信息
                audio_stream = None
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream:
                    sample_rate = int(audio_stream.get('sample_rate', 0))
                    channels = int(audio_stream.get('channels', 0))
                    duration = float(data['format'].get('duration', 0))
                    
                    # 简单的音频内容评估逻辑
                    needs_voice_extraction = False
                    reasons = []
                    
                    if channels > 1:
                        needs_voice_extraction = True
                        reasons.append("立体声音频，可能包含背景音乐")
                    
                    if duration > 10:
                        needs_voice_extraction = True
                        reasons.append("音频较长，建议分离背景音")
                    
                    return {
                        "success": True,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "duration": duration,
                        "needs_voice_extraction": needs_voice_extraction,
                        "reasons": reasons
                    }
            
            return {"success": False, "error": "无法分析音频内容"}
            
        except Exception as e:
            return {"success": False, "error": f"音频内容分析异常: {str(e)}"}