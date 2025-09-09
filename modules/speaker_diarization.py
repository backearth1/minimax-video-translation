import os
import librosa
import soundfile as sf
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json

class SpeakerDiarization:
    """
    说话人分离服务
    用于检测和分类多个说话人，避免音色克隆时混淆不同说话人的声音
    """
    
    def __init__(self, logger_service):
        self.logger = logger_service
        self.speaker_profiles = {}  # 存储说话人特征
        self.similarity_threshold = 0.40  # 声音相似度阈值（降低以提高说话人区分敏感度）
        
    def extract_voice_features(self, audio_path: str) -> np.ndarray:
        """提取增强的音频声纹特征"""
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 确保音频长度足够
            if len(y) < sr * 0.5:  # 少于0.5秒
                y = np.pad(y, (0, int(sr * 0.5 - len(y))), mode='constant')
            
            # 1. 增强的MFCC特征 (20维 + 一阶导数 + 二阶导数)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
            
            # 2. 基频特征 (F0) - 性别识别的关键特征
            f0 = librosa.yin(y, fmin=50, fmax=400, frame_length=1024)
            f0_valid = f0[f0 > 0]
            
            if len(f0_valid) > 0:
                f0_mean = np.mean(f0_valid)
                f0_std = np.std(f0_valid)
                f0_min = np.min(f0_valid)
                f0_max = np.max(f0_valid)
                f0_range = f0_max - f0_min
                f0_median = np.median(f0_valid)
            else:
                f0_mean = f0_std = f0_min = f0_max = f0_range = f0_median = 0
            
            # 3. 频谱特征 - 年龄和个体差异
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            
            # 4. 过零率 - 语音质量特征
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # 5. 梅尔频谱 - 额外的频谱信息
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
            mel_mean = np.mean(mel_spec, axis=1)
            
            # 6. 色度特征 - 音调模式
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # 7. 统计特征汇总
            spectral_features = np.array([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_flatness),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            # 8. 能量特征
            rms_energy = librosa.feature.rms(y=y)
            energy_features = np.array([
                np.mean(rms_energy),
                np.std(rms_energy),
                np.max(rms_energy),
                np.min(rms_energy)
            ])
            
            # 合并所有特征
            features = np.concatenate([
                mfcc_mean,           # 20维
                mfcc_std,            # 20维  
                mfcc_delta_mean,     # 20维
                mfcc_delta2_mean,    # 20维
                [f0_mean, f0_std, f0_min, f0_max, f0_range, f0_median],  # 6维
                spectral_features,   # 9维
                energy_features,     # 4维
                mel_mean,           # 13维
                chroma_mean         # 12维
            ])
            
            # 标准化特征
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.log("DEBUG", f"提取特征维度: {len(features)}维")
            return features
            
        except Exception as e:
            self.logger.log("ERROR", f"提取音频特征失败: {str(e)}")
            return np.zeros(124)  # 返回124维零向量作为备用
    
    def identify_speaker(self, audio_path: str, segment_id: int) -> str:
        """
        识别说话人身份
        返回说话人ID，如果是新说话人则创建新ID
        """
        try:
            # 提取当前音频的特征
            current_features = self.extract_voice_features(audio_path)
            
            if len(self.speaker_profiles) == 0:
                # 第一个说话人
                speaker_id = "speaker_1"
                self.speaker_profiles[speaker_id] = {
                    "features": current_features,
                    "segments": [segment_id],
                    "sample_count": 1
                }
                self.logger.log("INFO", f"检测到新说话人: {speaker_id}")
                return speaker_id
            
            # 与现有说话人比较相似度（使用加权方法）
            best_match = None
            highest_similarity = 0
            
            for speaker_id, profile in self.speaker_profiles.items():
                # 使用加权相似度计算
                similarity = self._calculate_weighted_similarity(current_features, profile["features"])
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = speaker_id
            
            # 判断是否为现有说话人
            if highest_similarity >= self.similarity_threshold:
                # 更新现有说话人特征（增量学习）
                profile = self.speaker_profiles[best_match]
                profile["features"] = (profile["features"] * profile["sample_count"] + current_features) / (profile["sample_count"] + 1)
                profile["segments"].append(segment_id)
                profile["sample_count"] += 1
                
                self.logger.log("INFO", f"匹配到说话人: {best_match} (相似度: {highest_similarity:.3f})")
                return best_match
            else:
                # 创建新说话人
                new_speaker_id = f"speaker_{len(self.speaker_profiles) + 1}"
                self.speaker_profiles[new_speaker_id] = {
                    "features": current_features,
                    "segments": [segment_id],
                    "sample_count": 1
                }
                self.logger.log("INFO", f"检测到新说话人: {new_speaker_id} (与最相似说话人相似度: {highest_similarity:.3f})")
                return new_speaker_id
                
        except Exception as e:
            self.logger.log("ERROR", f"说话人识别失败: {str(e)}")
            return f"speaker_unknown_{segment_id}"
    
    def get_speaker_representative_audio(self, speaker_id: str, segments: List[Dict]) -> str:
        """
        获取指定说话人的代表性音频片段
        选择时长较长且音质较好的片段作为音色克隆的样本
        """
        try:
            if speaker_id not in self.speaker_profiles:
                return None
            
            speaker_segments = []
            for segment in segments:
                if segment.get("speaker_id") == speaker_id:
                    speaker_segments.append(segment)
            
            if not speaker_segments:
                return None
            
            # 按音频时长排序，选择较长的片段
            speaker_segments.sort(key=lambda x: self._get_audio_duration(x.get("original_audio_path", "")), reverse=True)
            
            # 选择前3个最长的片段中音质最好的一个
            best_segment = None
            best_quality_score = 0
            
            for segment in speaker_segments[:3]:
                audio_path = segment.get("original_audio_path", "")
                if os.path.exists(audio_path):
                    quality_score = self._assess_audio_quality(audio_path)
                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_segment = segment
            
            if best_segment:
                self.logger.log("INFO", f"为说话人{speaker_id}选择代表音频: {best_segment['original_audio_path']}")
                return best_segment["original_audio_path"]
            
            return None
            
        except Exception as e:
            self.logger.log("ERROR", f"获取说话人代表音频失败: {str(e)}")
            return None
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            if not os.path.exists(audio_path):
                return 0.0
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except:
            return 0.0
    
    def _assess_audio_quality(self, audio_path: str) -> float:
        """评估音频质量"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 计算信噪比（简化版）
            rms_energy = librosa.feature.rms(y=y)[0]
            snr_estimate = np.mean(rms_energy) / (np.std(rms_energy) + 1e-8)
            
            # 计算频谱平坦度
            stft = librosa.stft(y)
            spectral_flatness = librosa.feature.spectral_flatness(S=np.abs(stft))
            flatness_score = 1 - np.mean(spectral_flatness)  # 越低越好
            
            # 综合质量评分
            quality_score = snr_estimate * 0.7 + flatness_score * 0.3
            return quality_score
            
        except:
            return 0.0
    
    def batch_analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        批量分析所有片段，检测并拆分包含多个说话人的片段
        """
        try:
            self.logger.log("INFO", "开始批量说话人分析和片段拆分...")
            
            processed_segments = []
            for segment in segments:
                audio_path = segment.get("original_audio_path", "")
                if os.path.exists(audio_path):
                    # 检测片段内是否有说话人变化
                    sub_segments = self.detect_speaker_changes_in_segment(segment)
                    processed_segments.extend(sub_segments)
                else:
                    segment["speaker_id"] = f"speaker_unknown_{segment['sequence']}"
                    processed_segments.append(segment)
            
            # 重新编号片段
            for i, seg in enumerate(processed_segments):
                seg["sequence"] = i + 1
            
            # 打印说话人统计信息
            speaker_stats = {}
            for segment in processed_segments:
                speaker_id = segment["speaker_id"]
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = 0
                speaker_stats[speaker_id] += 1
            
            self.logger.log("INFO", f"说话人分析完成，检测到{len(speaker_stats)}个说话人:")
            for speaker_id, count in speaker_stats.items():
                self.logger.log("INFO", f"  {speaker_id}: {count}个片段")
            
            self.logger.log("INFO", f"片段拆分完成: {len(segments)}个原始片段 → {len(processed_segments)}个处理后片段")
            
            return processed_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"批量说话人分析失败: {str(e)}")
            return segments
    
    def save_speaker_profiles(self, file_path: str = "./temp/speaker_profiles.json"):
        """保存说话人特征档案"""
        try:
            # 转换numpy数组为列表以便JSON序列化
            serializable_profiles = {}
            for speaker_id, profile in self.speaker_profiles.items():
                serializable_profiles[speaker_id] = {
                    "features": profile["features"].tolist(),
                    "segments": profile["segments"],
                    "sample_count": profile["sample_count"]
                }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_profiles, f, ensure_ascii=False, indent=2)
            
            self.logger.log("INFO", f"说话人特征档案已保存: {file_path}")
            
        except Exception as e:
            self.logger.log("ERROR", f"保存说话人特征档案失败: {str(e)}")
    
    def load_speaker_profiles(self, file_path: str = "./temp/speaker_profiles.json"):
        """加载说话人特征档案"""
        try:
            if not os.path.exists(file_path):
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                serializable_profiles = json.load(f)
            
            # 转换列表回numpy数组
            self.speaker_profiles = {}
            for speaker_id, profile in serializable_profiles.items():
                self.speaker_profiles[speaker_id] = {
                    "features": np.array(profile["features"]),
                    "segments": profile["segments"],
                    "sample_count": profile["sample_count"]
                }
            
            self.logger.log("INFO", f"说话人特征档案已加载: {len(self.speaker_profiles)}个说话人")
            
        except Exception as e:
            self.logger.log("ERROR", f"加载说话人特征档案失败: {str(e)}")
    
    def _calculate_weighted_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算加权相似度，重点关注性别和年龄相关特征"""
        try:
            # 特征分段权重
            # MFCC相关: 0-79维 (权重: 0.6)
            # 基频特征: 80-85维 (权重: 1.5 - 性别区分关键)  
            # 频谱特征: 86-94维 (权重: 1.0)
            # 能量特征: 95-98维 (权重: 0.8)
            # 梅尔频谱: 99-111维 (权重: 0.7)
            # 色度特征: 112-123维 (权重: 0.5)
            
            weights = np.ones(len(features1))
            
            # 提高基频特征权重（性别识别关键）
            weights[80:86] = 1.5
            
            # 提高频谱特征权重（个体差异）
            weights[86:95] = 1.2
            
            # 降低色度特征权重
            weights[112:124] = 0.5
            
            # 计算加权余弦相似度
            weighted_features1 = features1 * weights
            weighted_features2 = features2 * weights
            
            similarity = cosine_similarity(
                weighted_features1.reshape(1, -1),
                weighted_features2.reshape(1, -1)
            )[0][0]
            
            return similarity
            
        except Exception as e:
            self.logger.log("ERROR", f"计算加权相似度失败: {str(e)}")
            # fallback到标准余弦相似度
            return cosine_similarity(
                features1.reshape(1, -1),
                features2.reshape(1, -1)
            )[0][0]
    
    def detect_speaker_changes_in_segment(self, segment: Dict) -> List[Dict]:
        """
        检测单个音频片段内的说话人变化，如有变化则拆分
        """
        try:
            audio_path = segment.get("original_audio_path", "")
            if not os.path.exists(audio_path):
                segment["speaker_id"] = "unknown"
                return [segment]
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # 如果音频太短(<1秒)，不拆分
            if duration < 1.0:
                speaker_id = self.identify_speaker(audio_path, segment["sequence"])
                segment["speaker_id"] = speaker_id
                return [segment]
            
            # 滑动窗口分析 (每0.5秒一个窗口)
            window_size = int(0.5 * sr)  # 0.5秒窗口
            hop_size = int(0.25 * sr)    # 0.25秒跳跃
            
            features_sequence = []
            timestamps = []
            
            for start_sample in range(0, len(y) - window_size, hop_size):
                end_sample = start_sample + window_size
                window_audio = y[start_sample:end_sample]
                
                # 保存窗口音频到临时文件
                temp_window_path = f"./temp/window_{segment['sequence']}_{start_sample}.wav"
                sf.write(temp_window_path, window_audio, sr)
                
                # 提取特征
                features = self.extract_voice_features(temp_window_path)
                features_sequence.append(features)
                timestamps.append(start_sample / sr)
                
                # 清理临时文件
                if os.path.exists(temp_window_path):
                    os.remove(temp_window_path)
            
            # 检测说话人变化点
            change_points = self.find_speaker_change_points(features_sequence)
            
            if len(change_points) == 0:
                # 没有变化点，整个片段是同一个说话人
                speaker_id = self.identify_speaker(audio_path, segment["sequence"])
                segment["speaker_id"] = speaker_id
                return [segment]
            else:
                # 有变化点，拆分片段
                return self.split_segment_by_change_points(segment, change_points, timestamps, sr)
                
        except Exception as e:
            self.logger.log("ERROR", f"检测说话人变化失败: {str(e)}")
            # 失败时返回原片段
            segment["speaker_id"] = "unknown"
            return [segment]
    
    def find_speaker_change_points(self, features_sequence: List[np.ndarray]) -> List[int]:
        """
        找到说话人变化点
        """
        change_points = []
        change_threshold = 0.25  # 变化阈值，低于此相似度认为是不同说话人
        
        for i in range(1, len(features_sequence)):
            similarity = self._calculate_weighted_similarity(
                features_sequence[i-1], 
                features_sequence[i]
            )
            
            if similarity < change_threshold:
                change_points.append(i)
                self.logger.log("DEBUG", f"检测到说话人变化点: 位置{i}, 相似度{similarity:.3f}")
        
        return change_points
    
    def split_segment_by_change_points(self, original_segment: Dict, change_points: List[int], 
                                     timestamps: List[float], sr: int) -> List[Dict]:
        """
        根据变化点拆分音频片段
        """
        try:
            audio_path = original_segment["original_audio_path"]
            y, _ = librosa.load(audio_path, sr=sr)
            
            # 解析原始时间戳
            timestamp_str = original_segment["timestamp"]
            original_start, original_end = self._parse_timestamp_to_seconds(timestamp_str)
            
            segments = []
            last_point = 0
            
            # 根据变化点拆分
            for i, change_point in enumerate(change_points + [len(timestamps)]):
                if change_point <= last_point:
                    continue
                
                # 计算子片段的绝对时间
                start_time = original_start + timestamps[last_point]
                end_time = original_start + timestamps[min(change_point-1, len(timestamps)-1)] + 0.5
                
                # 提取子音频
                start_sample = int(timestamps[last_point] * sr)
                end_sample = int(min((timestamps[change_point-1] + 0.5), len(y)/sr) * sr) if change_point < len(timestamps) else len(y)
                
                sub_audio = y[start_sample:end_sample]
                
                # 保存子音频
                sub_audio_path = f"./temp/segment_{original_segment['sequence']}_part{i+1}.wav"
                sf.write(sub_audio_path, sub_audio, sr)
                
                # 识别说话人
                speaker_id = self.identify_speaker(sub_audio_path, original_segment['sequence'])
                
                # 创建新片段
                new_segment = original_segment.copy()
                new_segment.update({
                    "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                    "original_audio_path": sub_audio_path,
                    "speaker_id": speaker_id,
                    "translated_text": "",  # 重置翻译文本
                    "translated_audio_path": "",  # 重置翻译音频
                    "voice_id": ""  # 重置音色ID
                })
                
                segments.append(new_segment)
                last_point = change_point
            
            self.logger.log("INFO", f"片段{original_segment['sequence']}拆分为{len(segments)}个子片段")
            return segments
            
        except Exception as e:
            self.logger.log("ERROR", f"拆分片段失败: {str(e)}")
            return [original_segment]
    
    def _parse_timestamp_to_seconds(self, timestamp: str) -> tuple:
        """解析时间戳字符串为秒数"""
        try:
            parts = timestamp.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                return start, end
        except:
            pass
        return 0.0, 3.0