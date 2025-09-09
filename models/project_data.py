import re
from datetime import datetime
from typing import List, Dict, Any

class ProjectData:
    def __init__(self):
        self.video_path = None
        self.video_filename = None
        self.final_video_path = None
        self.segments = []
        self.processing_status = "idle"  # idle, processing, completed, error
        self.current_step = ""
        self.progress = 0
        
    def set_video_path(self, path: str):
        self.video_path = path
        self.video_filename = path.split('/')[-1]
        
    def add_segment(self, sequence: int, timestamp: str, original_text: str, 
                   translated_text: str = "", original_audio_path: str = "",
                   translated_audio_path: str = "", voice_id: str = "", speed: float = 1.0):
        segment = {
            "sequence": sequence,
            "timestamp": timestamp,
            "original_text": original_text,
            "translated_text": translated_text,
            "original_audio_path": original_audio_path,
            "translated_audio_path": translated_audio_path,
            "voice_id": voice_id,
            "speed": speed
        }
        self.segments.append(segment)
        
    def update_segment(self, sequence: int, **kwargs):
        for segment in self.segments:
            if segment["sequence"] == sequence:
                segment.update(kwargs)
                break
                
    def get_segment(self, sequence: int):
        for segment in self.segments:
            if segment["sequence"] == sequence:
                return segment
        return None
        
    def update_segments(self, segments_data: List[Dict]):
        self.segments = segments_data
        
    def clear_segments(self):
        self.segments = []
        
    def set_processing_status(self, status: str, step: str = "", progress: int = 0):
        self.processing_status = status
        self.current_step = step
        self.progress = progress
        
    def export_srt(self) -> str:
        """导出SRT字幕文件格式"""
        srt_content = []
        
        for i, segment in enumerate(self.segments, 1):
            # 解析时间戳
            start_time, end_time = self._parse_timestamp(segment["timestamp"])
            
            # 格式化时间为SRT格式 (HH:MM:SS,mmm)
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # 构建SRT条目
            srt_entry = f"{i}\n{start_srt} --> {end_srt}\n{segment['translated_text']}\n"
            srt_content.append(srt_entry)
            
        return "\n".join(srt_content)
    
    def import_srt(self, srt_content: str) -> List[Dict]:
        """导入SRT字幕文件"""
        segments = []
        
        # 清理内容并按段落分割
        srt_blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in srt_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    sequence = int(lines[0])
                    time_line = lines[1]
                    text = '\n'.join(lines[2:])
                    
                    # 解析时间戳 "00:00:01,500 --> 00:00:04,200"
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
                    if time_match:
                        start_srt, end_srt = time_match.groups()
                        start_seconds = self._srt_time_to_seconds(start_srt)
                        end_seconds = self._srt_time_to_seconds(end_srt)
                        timestamp = f"{start_seconds:.2f}-{end_seconds:.2f}"
                        
                        segment = {
                            "sequence": sequence,
                            "timestamp": timestamp,
                            "original_text": "",
                            "translated_text": text,
                            "original_audio_path": "",
                            "translated_audio_path": "",
                            "voice_id": "",
                            "speed": 1.0
                        }
                        segments.append(segment)
                        
                except (ValueError, IndexError):
                    continue
                    
        self.segments = segments
        return segments
    
    def _parse_timestamp(self, timestamp: str) -> tuple:
        """解析 '00:00-00:03' 或 '0.0-3.0' 格式的时间戳"""
        parts = timestamp.split('-')
        if len(parts) != 2:
            return 0.0, 0.0
            
        start_str, end_str = parts
        
        try:
            # 尝试解析秒数格式
            start_time = float(start_str)
            end_time = float(end_str)
        except ValueError:
            try:
                # 尝试解析时分秒格式 MM:SS
                start_time = self._time_to_seconds(start_str)
                end_time = self._time_to_seconds(end_str)
            except:
                return 0.0, 0.0
                
        return start_time, end_time
    
    def _time_to_seconds(self, time_str: str) -> float:
        """将 MM:SS 或 HH:MM:SS 转换为秒数"""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return float(time_str)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """将秒数转换为SRT时间格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """将SRT时间格式转换为秒数"""
        # 格式: HH:MM:SS,mmm
        time_part, millis_part = srt_time.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        millis = int(millis_part)
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        return total_seconds
    
    def get_total_duration(self) -> float:
        """获取总时长（秒）"""
        if not self.segments:
            return 0.0
            
        max_end_time = 0.0
        for segment in self.segments:
            _, end_time = self._parse_timestamp(segment["timestamp"])
            max_end_time = max(max_end_time, end_time)
            
        return max_end_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_filename": self.video_filename,
            "segments": self.segments,
            "processing_status": self.processing_status,
            "current_step": self.current_step,
            "progress": self.progress,
            "total_duration": self.get_total_duration(),
            "segment_count": len(self.segments)
        }