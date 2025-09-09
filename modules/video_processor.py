import os
import subprocess
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple

class VideoProcessor:
    def __init__(self, logger_service):
        self.logger = logger_service
        
    def extract_audio(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """从视频中提取音频"""
        try:
            if not output_path:
                video_name = Path(video_path).stem
                output_path = f"./temp/{video_name}_audio.wav"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 不要视频流
                '-acodec', 'pcm_s16le',  # 音频编码
                '-ar', '16000',  # 采样率16kHz
                '-ac', '1',  # 单声道
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            self.logger.log("INFO", f"开始提取音频: {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # 获取音频信息
                duration = self.get_audio_duration(output_path)
                file_size = os.path.getsize(output_path)
                
                self.logger.log("INFO", f"音频提取成功: 时长{duration:.2f}s, 大小{file_size/1024/1024:.2f}MB")
                
                return {
                    "success": True,
                    "audio_path": output_path,
                    "duration": duration,
                    "file_size": file_size
                }
            else:
                error_msg = f"ffmpeg错误: {result.stderr}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "音频提取超时"}
        except Exception as e:
            error_msg = f"音频提取异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"success": False, "error": "无法打开视频文件"}
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            file_size = os.path.getsize(video_path)
            
            return {
                "success": True,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "file_size": file_size
            }
            
        except Exception as e:
            return {"success": False, "error": f"获取视频信息失败: {str(e)}"}
    
    def get_audio_duration(self, audio_path: str) -> float:
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
    
    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        """合并音频和视频"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cmd = [
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy',  # 复制视频流
                '-c:a', 'aac',   # 音频编码为AAC
                '-strict', 'experimental',
                '-map', '0:v:0', '-map', '1:a:0',  # 映射视频和音频流
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            self.logger.log("INFO", f"开始合并音频和视频...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                file_size = os.path.getsize(output_path)
                self.logger.log("INFO", f"音视频合并成功: {output_path} ({file_size/1024/1024:.2f}MB)")
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size
                }
            else:
                error_msg = f"合并失败: {result.stderr}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "音视频合并超时"}
        except Exception as e:
            error_msg = f"合并异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}