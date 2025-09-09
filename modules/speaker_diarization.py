import os
import librosa
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
        self.similarity_threshold = 0.75  # 声音相似度阈值
        
    def extract_voice_features(self, audio_path: str) -> np.ndarray:
        """提取音频的声音特征"""
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 提取MFCC特征（梅尔频率倒谱系数）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # 提取音调特征
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # 提取频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # 提取频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            
            # 合并所有特征
            features = np.concatenate([
                mfcc_mean,
                [pitch_mean, spectral_centroid_mean, spectral_bandwidth_mean]
            ])
            
            return features
            
        except Exception as e:
            self.logger.log("ERROR", f"提取音频特征失败: {str(e)}")
            return np.zeros(16)  # 返回零向量作为备用
    
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
            
            # 与现有说话人比较相似度
            best_match = None
            highest_similarity = 0
            
            for speaker_id, profile in self.speaker_profiles.items():
                similarity = cosine_similarity(
                    current_features.reshape(1, -1),
                    profile["features"].reshape(1, -1)
                )[0][0]
                
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
        批量分析所有片段的说话人身份
        """
        try:
            self.logger.log("INFO", "开始批量说话人分析...")
            
            processed_segments = []
            for segment in segments:
                audio_path = segment.get("original_audio_path", "")
                if os.path.exists(audio_path):
                    speaker_id = self.identify_speaker(audio_path, segment["sequence"])
                    segment["speaker_id"] = speaker_id
                else:
                    segment["speaker_id"] = f"speaker_unknown_{segment['sequence']}"
                
                processed_segments.append(segment)
            
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