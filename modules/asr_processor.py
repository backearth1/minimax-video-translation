import os
import whisper
import librosa
import numpy as np
from typing import List, Dict, Any, Tuple

class ASRProcessor:
    def __init__(self, config, logger_service):
        self.config = config
        self.logger = logger_service
        self.model = None
        
    def load_model(self):
        """加载Whisper模型"""
        if self.model is None:
            try:
                self.logger.log("INFO", "加载Whisper模型...")
                self.model = whisper.load_model("base")
                self.logger.log("INFO", "Whisper模型加载完成")
            except Exception as e:
                self.logger.log("ERROR", f"Whisper模型加载失败: {str(e)}")
                raise e
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """语音识别"""
        try:
            self.load_model()
            
            self.logger.log("INFO", f"开始ASR语音识别: {audio_path}")
            
            # 将配置的原语言转换为Whisper语言代码
            language_code = self._get_whisper_language_code(self.config.source_language)
            self.logger.log("INFO", f"使用语言代码进行ASR识别: {self.config.source_language} -> {language_code}")
            
            # 使用Whisper进行识别
            result = self.model.transcribe(
                audio_path,
                language=language_code,
                word_timestamps=True,  # 启用词级时间戳
                verbose=False
            )
            
            # 提取分段信息
            segments = []
            for segment in result['segments']:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'words': segment.get('words', [])
                })
            
            self.logger.log("INFO", f"ASR识别完成，检测到{len(segments)}个原始片段")
            
            return {
                "success": True,
                "text": result['text'],
                "segments": segments,
                "language": result.get('language', 'zh')
            }
            
        except Exception as e:
            error_msg = f"ASR识别失败: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def smart_segment_split(self, segments: List[Dict], audio_path: str) -> List[Dict]:
        """智能切分语音片段"""
        try:
            self.logger.log("INFO", "开始智能切分语音片段...")
            
            # 获取配置参数
            min_duration = self.config.min_segment_duration
            max_duration = self.config.max_segment_duration
            split_mode = self.config.asr_split_mode
            
            # 根据模式调整参数
            if split_mode == "保守模式":
                target_duration = 5.0
                merge_threshold = 2.0
            elif split_mode == "激进模式":
                target_duration = 2.5
                merge_threshold = 1.0
            else:  # 平衡模式
                target_duration = 3.5
                merge_threshold = 1.5
            
            optimized_segments = []
            current_segment = None
            sequence = 1
            
            for segment in segments:
                duration = segment['end'] - segment['start']
                text = segment['text']
                
                # 如果当前片段太短，尝试与下一个合并
                if duration < min_duration and current_segment is None:
                    current_segment = segment.copy()
                    continue
                
                # 如果有待合并的片段
                if current_segment is not None:
                    # 检查合并后的时长
                    merged_duration = segment['end'] - current_segment['start']
                    
                    if merged_duration <= max_duration:
                        # 合并片段
                        current_segment['end'] = segment['end']
                        current_segment['text'] += segment['text']
                        
                        # 如果合并后达到目标时长，输出片段
                        if merged_duration >= target_duration:
                            optimized_segments.append(self._create_segment(
                                current_segment, sequence, audio_path
                            ))
                            sequence += 1
                            current_segment = None
                    else:
                        # 输出当前片段，开始新片段
                        optimized_segments.append(self._create_segment(
                            current_segment, sequence, audio_path
                        ))
                        sequence += 1
                        current_segment = segment.copy()
                else:
                    # 检查片段是否需要拆分
                    if duration > max_duration:
                        # 拆分长片段
                        split_segments = self._split_long_segment(segment, max_duration)
                        for split_seg in split_segments:
                            optimized_segments.append(self._create_segment(
                                split_seg, sequence, audio_path
                            ))
                            sequence += 1
                    elif duration >= min_duration:
                        # 直接使用合适的片段
                        optimized_segments.append(self._create_segment(
                            segment, sequence, audio_path
                        ))
                        sequence += 1
                    else:
                        # 太短的片段暂存
                        current_segment = segment.copy()
            
            # 处理最后的待合并片段
            if current_segment is not None:
                optimized_segments.append(self._create_segment(
                    current_segment, sequence, audio_path
                ))
            
            self.logger.log("INFO", f"智能切分完成，优化后得到{len(optimized_segments)}个片段")
            
            return optimized_segments
            
        except Exception as e:
            error_msg = f"智能切分失败: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return []
    
    def _create_segment(self, segment: Dict, sequence: int, audio_path: str) -> Dict:
        """创建标准片段数据结构"""
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # 提取音频片段
        segment_audio_path = self._extract_audio_segment(
            audio_path, start_time, end_time, sequence
        )
        
        return {
            "sequence": sequence,
            "timestamp": f"{start_time:.2f}-{end_time:.2f}",
            "original_text": text,
            "translated_text": "",
            "original_audio_path": segment_audio_path,
            "translated_audio_path": "",
            "voice_id": "",
            "speed": 1.0
        }
    
    def _split_long_segment(self, segment: Dict, max_duration: float) -> List[Dict]:
        """拆分过长的片段"""
        duration = segment['end'] - segment['start']
        text = segment['text']
        
        # 简单的时间均分策略
        num_splits = int(np.ceil(duration / max_duration))
        split_duration = duration / num_splits
        
        splits = []
        for i in range(num_splits):
            start = segment['start'] + i * split_duration
            end = min(segment['start'] + (i + 1) * split_duration, segment['end'])
            
            # 简单的文本分割（按字符数量）
            text_start = int(i * len(text) / num_splits)
            text_end = int((i + 1) * len(text) / num_splits)
            split_text = text[text_start:text_end].strip()
            
            splits.append({
                'start': start,
                'end': end,
                'text': split_text,
                'words': []
            })
        
        return splits
    
    def _extract_audio_segment(self, audio_path: str, start_time: float, 
                              end_time: float, sequence: int) -> str:
        """提取音频片段"""
        try:
            output_path = f"./temp/segment_{sequence}_original.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 使用librosa提取音频片段
            y, sr = librosa.load(audio_path, sr=16000)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            segment_audio = y[start_sample:end_sample]
            
            # 保存音频片段
            import soundfile as sf
            sf.write(output_path, segment_audio, sr)
            
            return output_path
            
        except Exception as e:
            self.logger.log("ERROR", f"提取音频片段失败: {str(e)}")
            return ""
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """完整的音频处理流程"""
        try:
            # 1. ASR识别
            asr_result = self.transcribe_audio(audio_path)
            if not asr_result["success"]:
                return []
            
            # 2. 智能切分
            segments = self.smart_segment_split(asr_result["segments"], audio_path)
            
            return segments
            
        except Exception as e:
            self.logger.log("ERROR", f"音频处理失败: {str(e)}")
            return []
    
    def _get_whisper_language_code(self, language_name: str) -> str:
        """将配置的语言名称转换为Whisper语言代码"""
        language_mapping = {
            "中文": "zh",
            "粤语": "yue", 
            "英语": "en",
            "西班牙语": "es",
            "法语": "fr",
            "俄语": "ru",
            "德语": "de",
            "葡萄牙语": "pt",
            "阿拉伯语": "ar",
            "意大利语": "it",
            "日语": "ja",
            "韩语": "ko",
            "印尼语": "id",
            "越南语": "vi",
            "土耳其语": "tr",
            "荷兰语": "nl",
            "乌克兰语": "uk",
            "泰语": "th",
            "波兰语": "pl",
            "罗马尼亚语": "ro",
            "希腊语": "el",
            "捷克语": "cs",
            "芬兰语": "fi",
            "印地语": "hi",
            "保加利亚语": "bg",
            "丹麦语": "da",
            "希伯来语": "he",
            "马来语": "ms",
            "波斯语": "fa",
            "斯洛伐克语": "sk",
            "瑞典语": "sv",
            "克罗地亚语": "hr",
            "菲律宾语": "tl",
            "匈牙利语": "hu",
            "挪威语": "no",
            "斯洛文尼亚语": "sl",
            "加泰罗尼亚语": "ca",
            "尼诺斯克语": "nn",
            "泰米尔语": "ta",
            "阿非利卡语": "af"
        }
        
        return language_mapping.get(language_name, "zh")  # 默认返回中文