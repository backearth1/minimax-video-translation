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
        """合并音频和视频，完全替换原始音频"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 第一步：彻底移除原始音频轨道
            silent_video_path = output_path.replace('.mp4', '_silent.mp4')
            
            self.logger.log("INFO", "第一步：彻底移除原始音频轨道...")
            # 使用更强的参数确保完全移除音频
            remove_audio_cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'copy',     # 复制视频流，保持原始质量
                '-an',              # 移除所有音频流
                '-map', '0:v',      # 只映射视频流
                '-avoid_negative_ts', 'make_zero',  # 避免时间戳问题
                '-y',               # 覆盖输出文件
                silent_video_path
            ]
            
            result1 = subprocess.run(remove_audio_cmd, capture_output=True, text=True, timeout=300)
            
            if result1.returncode != 0:
                error_msg = f"移除原始音频失败: {result1.stderr}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
            
            self.logger.log("INFO", "第二步：添加新的翻译音频...")
            
            # 第二步：添加新音频到静音视频，确保完全替换
            final_cmd = [
                'ffmpeg', '-i', silent_video_path, '-i', audio_path,
                '-c:v', 'copy',              # 复制视频流，保持原始质量
                '-c:a', 'aac',               # 音频编码为AAC
                '-b:a', '192k',              # 提高音频比特率到192k，确保质量
                '-ar', '44100',              # 设置音频采样率
                '-ac', '2',                  # 立体声
                '-map', '0:v:0',             # 映射静音视频的视频流
                '-map', '1:a:0',             # 映射新音频流
                '-avoid_negative_ts', 'make_zero',  # 避免时间戳问题
                '-shortest',                 # 使用最短流的长度
                '-movflags', '+faststart',   # 优化播放性能
                '-y',                        # 覆盖输出文件
                output_path
            ]
            
            result2 = subprocess.run(final_cmd, capture_output=True, text=True, timeout=600)
            
            # 清理临时文件
            if os.path.exists(silent_video_path):
                os.remove(silent_video_path)
            
            if result2.returncode == 0:
                file_size = os.path.getsize(output_path)
                self.logger.log("INFO", f"音视频合并成功: {output_path} ({file_size/1024/1024:.2f}MB)")
                self.logger.log("INFO", "✅ 原始音频已完全替换为翻译音频")
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size
                }
            else:
                error_msg = f"添加新音频失败: {result2.stderr}"
                self.logger.log("ERROR", error_msg)
                return {"success": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "音视频合并超时"}
        except Exception as e:
            error_msg = f"合并异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            # 清理可能的临时文件
            silent_video_path = output_path.replace('.mp4', '_silent.mp4')
            if os.path.exists(silent_video_path):
                os.remove(silent_video_path)
            return {"success": False, "error": error_msg}