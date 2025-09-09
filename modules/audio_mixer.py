import os
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any

class AudioMixer:
    def __init__(self, logger_service):
        self.logger = logger_service
        
    def concatenate_audio_segments(self, segments: List[Dict], output_path: str) -> Dict[str, Any]:
        """使用绝对时间戳拼接音频片段"""
        try:
            self.logger.log("INFO", f"开始按绝对时间戳拼接{len(segments)}个音频片段...")
            
            # 目标采样率
            target_sr = 44100
            
            # 找到最大结束时间以确定总音频长度
            max_end_time = 0
            for segment in segments:
                timestamp = segment.get('timestamp', '0-3')
                start_time, end_time = self._parse_timestamp(timestamp)
                max_end_time = max(max_end_time, end_time)
            
            # 创建总音频数组（填充静音）
            total_samples = int(max_end_time * target_sr)
            final_audio = np.zeros(total_samples)
            
            self.logger.log("INFO", f"创建总时长{max_end_time:.2f}s的音频轨道")
            
            for i, segment in enumerate(segments):
                try:
                    sequence = segment.get('sequence', i+1)
                    translated_audio_path = segment.get('translated_audio_path', '')
                    timestamp = segment.get('timestamp', '0-3')
                    
                    # 解析绝对时间戳
                    start_time, end_time = self._parse_timestamp(timestamp)
                    start_sample = int(start_time * target_sr)
                    end_sample = int(end_time * target_sr)
                    expected_samples = end_sample - start_sample
                    
                    if translated_audio_path and os.path.exists(translated_audio_path):
                        # 加载翻译音频
                        audio, sr = librosa.load(translated_audio_path, sr=target_sr)
                        
                        # 调整音频长度到期望时长
                        if len(audio) > expected_samples:
                            # 音频太长，裁剪
                            audio = audio[:expected_samples]
                        elif len(audio) < expected_samples:
                            # 音频太短，填充静音
                            padding = expected_samples - len(audio)
                            audio = np.concatenate([audio, np.zeros(padding)])
                        
                        # 将音频放置到绝对位置
                        final_audio[start_sample:start_sample + len(audio)] = audio
                        
                        self.logger.log("INFO", f"第{sequence}句音频已放置到 {start_time:.2f}s-{end_time:.2f}s 位置")
                        
                    else:
                        self.logger.log("WARNING", f"第{sequence}句没有翻译音频，保持静音: {start_time:.2f}s-{end_time:.2f}s")
                    
                except Exception as e:
                    self.logger.log("ERROR", f"处理第{sequence}句音频时出错: {str(e)}")
            
            # 保存最终音频
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, final_audio, target_sr)
            
            duration = len(final_audio) / target_sr
            file_size = os.path.getsize(output_path)
            
            self.logger.log("INFO", f"绝对时间戳音频拼接完成: 总时长{duration:.2f}s, 大小{file_size/1024/1024:.2f}MB")
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": duration,
                "file_size": file_size
            }
            
        except Exception as e:
            error_msg = f"音频拼接失败: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def mix_with_background(self, translated_audio_path: str, background_audio_path: str, 
                           output_path: str, background_volume: float = 0.3) -> Dict[str, Any]:
        """将翻译音频与背景音乐混合"""
        try:
            if not os.path.exists(background_audio_path):
                # 没有背景音频，直接复制翻译音频
                import shutil
                shutil.copy2(translated_audio_path, output_path)
                return {"success": True, "output_path": output_path}
            
            self.logger.log("INFO", "开始混合翻译音频和背景音乐...")
            
            # 加载音频
            translated_audio, sr1 = librosa.load(translated_audio_path, sr=44100)
            background_audio, sr2 = librosa.load(background_audio_path, sr=44100)
            
            # 调整背景音频长度
            if len(background_audio) > len(translated_audio):
                background_audio = background_audio[:len(translated_audio)]
            elif len(background_audio) < len(translated_audio):
                # 循环播放背景音频
                repeat_times = int(np.ceil(len(translated_audio) / len(background_audio)))
                background_audio = np.tile(background_audio, repeat_times)[:len(translated_audio)]
            
            # 降低背景音乐音量
            background_audio = background_audio * background_volume
            
            # 混合音频
            mixed_audio = translated_audio + background_audio
            
            # 防止音频溢出
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val * 0.95
            
            # 保存混合音频
            sf.write(output_path, mixed_audio, 44100)
            
            self.logger.log("INFO", f"音频混合完成: {output_path}")
            
            return {"success": True, "output_path": output_path}
            
        except Exception as e:
            error_msg = f"音频混合失败: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def _parse_timestamp(self, timestamp: str) -> tuple:
        """解析时间戳"""
        try:
            parts = timestamp.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                return start, end
        except:
            pass
        return 0.0, 3.0