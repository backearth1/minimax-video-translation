import os
import torch
import torchaudio
import whisper_timestamped as whisper
from pyannote.audio import Pipeline
from typing import Dict, Any, List, Tuple
import librosa
import soundfile as sf
import subprocess
import tempfile
from pathlib import Path
from .model_manager import ModelManager

class ProfessionalAudioProcessor:
    """
    专业音频处理器
    集成 Demucs + pyannote.audio + whisper-timestamped
    提供工业级音频分离和精确ASR分割
    """
    
    def __init__(self, logger_service):
        self.logger = logger_service
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型管理器
        self.model_manager = ModelManager(logger_service)
        
        # 延迟初始化模型 - 只在实际使用时加载
        self.whisper_model = None
        self.diarization_pipeline = None
        self._models_initialized = False
        self.recommended_config = {}
        
        self.logger.log("INFO", "专业音频处理器创建完成 (模型将在首次使用时加载)")
    
    def check_models_status(self):
        """检查并显示所有模型状态 - 用户可主动调用"""
        self.logger.log("INFO", "🔍 检查专业音频处理模型状态...")
        self.model_manager.print_model_status()
        
        # 提供下载建议
        status = self.model_manager.check_model_availability()
        missing_any = any(not info["available"] for info in status.values())
        
        if missing_any:
            self.logger.log("INFO", "💡 建议:")
            self.logger.log("INFO", "   1. 首次使用专业AI翻译时，系统会自动下载缺失模型")
            self.logger.log("INFO", "   2. 建议在网络良好时进行首次处理")
            self.logger.log("INFO", "   3. 所有模型仅需下载一次，后续使用无需重复下载")
        else:
            self.logger.log("INFO", "✅ 所有模型已准备就绪，可直接使用专业AI翻译")
    
    def _check_system_resources(self):
        """检查系统资源状况"""
        try:
            import psutil
            
            # 获取内存信息
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            self.logger.log("INFO", f"系统可用内存: {available_gb:.1f}GB")
            
            # 内存不足警告
            if available_gb < 2:
                self.logger.log("WARNING", "系统可用内存不足2GB，可能导致模型加载失败")
                return False
            elif available_gb < 4:
                self.logger.log("WARNING", "系统可用内存较低，建议使用轻量级模型")
                
            return True
            
        except ImportError:
            self.logger.log("WARNING", "无法检查系统资源 (psutil未安装)")
            return True
        except Exception as e:
            self.logger.log("WARNING", f"系统资源检查失败: {str(e)}")
            return True
    
    def _initialize_models(self):
        """初始化所有AI模型"""
        try:
            self.logger.log("INFO", "正在初始化专业音频处理模型...")
            
            # 0. 检查系统资源和模型可用性
            if not self._check_system_resources():
                self.logger.log("ERROR", "系统资源不足，跳过模型初始化")
                return
            
            # 获取推荐的模型配置
            self.recommended_config = self.model_manager.prepare_models_for_professional_processing()
            
            # 1. 初始化 Whisper-timestamped (根据推荐配置)
            try:
                recommended_whisper = self.recommended_config.get("whisper", "base")
                self.logger.log("INFO", f"加载 Whisper-timestamped 模型: {recommended_whisper}")
                
                # 检查模型是否已缓存
                status = self.model_manager.check_model_availability()
                if not status["whisper"]["available"] or recommended_whisper not in status["whisper"]["cached_models"]:
                    estimate = self.model_manager.estimate_download_time("whisper").get(recommended_whisper, "未知")
                    self.logger.log("INFO", f"🌐 首次下载 {recommended_whisper} 模型，预计耗时: {estimate}")
                
                # 尝试加载推荐模型，失败则回退
                model_priority = [recommended_whisper, "base"] if recommended_whisper != "base" else ["base"]
                
                for model_name in model_priority:
                    try:
                        self.logger.log("INFO", f"尝试加载 {model_name} 模型...")
                        
                        # 添加内存检查
                        if self.device.type == "cuda":
                            import torch
                            torch.cuda.empty_cache()  # 清理GPU缓存
                        
                        # 使用项目模型目录
                        project_model_dir = self.model_manager.models_dir
                        whisper_model_path = os.path.join(project_model_dir, "whisper", f"{model_name}.pt")
                        
                        if os.path.exists(whisper_model_path):
                            # 从项目目录加载模型
                            self.whisper_model = whisper.load_model(whisper_model_path, device=self.device)
                        else:
                            # 回退到标准加载（会触发下载）
                            self.whisper_model = whisper.load_model(model_name, device=self.device)
                        self.logger.log("INFO", f"✅ Whisper {model_name} 模型加载成功")
                        break
                    except Exception as model_err:
                        self.logger.log("WARNING", f"{model_name} 模型加载失败: {str(model_err)}")
                        # 尝试释放内存
                        if hasattr(self, 'whisper_model') and self.whisper_model:
                            del self.whisper_model
                            self.whisper_model = None
                        continue
                
                if not self.whisper_model:
                    raise Exception("所有 Whisper 模型加载失败")
                    
            except Exception as e:
                self.logger.log("ERROR", f"Whisper 模型加载失败: {str(e)}")
                self.whisper_model = None
            
            # 2. 初始化 pyannote.audio (根据推荐配置)
            try:
                recommended_pyannote = self.recommended_config.get("pyannote", "pyannote/speaker-diarization-3.1")
                self.logger.log("INFO", f"加载 pyannote.audio 模型: {recommended_pyannote}")
                
                # 检查模型是否已缓存
                if not status["pyannote"]["available"]:
                    estimate = self.model_manager.estimate_download_time("pyannote").get(recommended_pyannote, "2-3分钟")
                    self.logger.log("INFO", f"🌐 首次下载 pyannote.audio 模型，预计耗时: {estimate}")
                
                # 使用环境变量或配置文件中的HuggingFace token
                auth_token = os.getenv("HUGGINGFACE_TOKEN", None)
                
                # 设置HF_HOME指向项目目录，让pyannote从项目目录加载模型
                old_hf_home = os.environ.get("HF_HOME", None)
                os.environ["HF_HOME"] = os.path.join(self.model_manager.models_dir, "pyannote")
                
                try:
                    # 使用标准模型名加载（会从HF_HOME查找）
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        recommended_pyannote,
                        use_auth_token=auth_token
                    )
                    
                    self.diarization_pipeline = self.diarization_pipeline.to(self.device)
                    self.logger.log("INFO", "✅ pyannote.audio 模型加载成功")
                except Exception as load_err:
                    self.logger.log("WARNING", f"pyannote.audio 模型加载失败: {str(load_err)}")
                    # 如果token有问题，尝试无token加载
                    if "token" in str(load_err).lower() or "unauthorized" in str(load_err).lower():
                        self.logger.log("INFO", "尝试无token加载pyannote.audio...")
                        try:
                            self.diarization_pipeline = Pipeline.from_pretrained(recommended_pyannote)
                            self.diarization_pipeline = self.diarization_pipeline.to(self.device)
                            self.logger.log("INFO", "✅ pyannote.audio 模型加载成功(无token)")
                        except Exception as e2:
                            self.logger.log("ERROR", f"无token加载也失败: {str(e2)}")
                            raise load_err
                finally:
                    # 恢复原始HF_HOME环境变量
                    if old_hf_home is not None:
                        os.environ["HF_HOME"] = old_hf_home
                    elif "HF_HOME" in os.environ:
                        del os.environ["HF_HOME"]
                    
            except Exception as e:
                self.logger.log("ERROR", f"pyannote.audio 加载失败: {str(e)}")
                self.diarization_pipeline = None
                
                # 确保恢复环境变量
                if 'old_hf_home' in locals():
                    if old_hf_home is not None:
                        os.environ["HF_HOME"] = old_hf_home
                    elif "HF_HOME" in os.environ:
                        del os.environ["HF_HOME"]
            
            self.logger.log("INFO", f"🚀 专业音频处理器初始化完成 (设备: {self.device})")
            
        except Exception as e:
            self.logger.log("ERROR", f"专业音频处理器初始化失败: {str(e)}")
    
    def process_audio_professionally(self, audio_path: str, source_language: str = "zh", project_data=None) -> Dict[str, Any]:
        """
        专业音频处理主流程
        
        Args:
            audio_path: 原始音频路径
            source_language: 源语言代码
            
        Returns:
            处理结果字典
        """
        try:
            self.logger.log("INFO", "🎵 开始专业音频处理流程...")
            
            # 首次使用时初始化模型
            if not self._models_initialized:
                self._initialize_models()
                self._models_initialized = True
            
            # 步骤1: Demucs 音频源分离
            separation_result = self._separate_audio_sources(audio_path)
            if not separation_result["success"]:
                return separation_result
            
            vocals_path = separation_result["vocals_path"]
            background_path = separation_result["background_path"]
            
            # 立即更新project_data以便前端预览
            if project_data:
                project_data.vocals_audio_path = vocals_path
                project_data.background_audio_path = background_path
                project_data.set_processing_status("processing", "🎵 音频分离完成，开始说话人分析...", 30)
                self.logger.log("INFO", "✅ Demucs分离完成，音频预览已更新")
            
            # 步骤2: 智能语音识别+VAD切分 (30% -> 60%)
            if project_data:
                project_data.set_processing_status("processing", "🗣️ 智能语音识别和切分...", 30)
            self.logger.log("INFO", "📝 开始智能语音识别...")
            word_timestamps = self._transcribe_with_timestamps(vocals_path, source_language)
            
            # 步骤3: 基于停顿的语义切分 (60% -> 70%)
            if project_data:
                project_data.set_processing_status("processing", "✂️ 智能语义片段切分...", 60)
            self.logger.log("INFO", "✂️ 创建语义片段...")
            semantic_segments = self._create_semantic_segments(word_timestamps)
            
            # 步骤4: 说话人分离分析 (70% -> 80%)
            if project_data:
                project_data.set_processing_status("processing", "🎭 分析说话人分布...", 70)
            self.logger.log("INFO", "📊 开始说话人分离分析...")
            speaker_segments = self._analyze_speakers(vocals_path)
            
            # 步骤5: 片段级说话人检测 (80% -> 90%)
            if project_data:
                project_data.set_processing_status("processing", "🔍 检测多说话人片段...", 80)
            self.logger.log("INFO", "🔍 检测多说话人片段...")
            aligned_segments = self._detect_multi_speaker_segments(semantic_segments, speaker_segments)
            
            # 步骤5: 生成最终片段
            final_segments = self._generate_audio_segments(vocals_path, aligned_segments)
            
            self.logger.log("INFO", f"✅ 专业音频处理完成: {len(final_segments)}个精确片段")
            
            return {
                "success": True,
                "vocals_path": vocals_path,
                "background_path": background_path,
                "segments": final_segments,
                "total_segments": len(final_segments)
            }
            
        except Exception as e:
            error_msg = f"专业音频处理失败: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def _separate_audio_sources(self, audio_path: str) -> Dict[str, Any]:
        """使用 Demucs 进行音频源分离"""
        try:
            self.logger.log("INFO", "🎼 使用 Demucs 进行音频源分离...")
            
            # 检查Demucs模型状态
            status = self.model_manager.check_model_availability()
            if not status["demucs"]["available"]:
                estimate = self.model_manager.estimate_download_time("demucs").get("htdemucs", "3-5分钟")
                self.logger.log("INFO", f"🌐 首次使用 Demucs，可能需要下载模型，预计耗时: {estimate}")
            
            # 创建临时输出目录
            output_dir = "./temp/demucs_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 运行 Demucs 分离 (使用UV环境，指定项目模型)
            # 设置环境变量指向项目模型目录
            env = os.environ.copy()
            env["TORCH_HOME"] = os.path.join(self.model_manager.models_dir, "demucs")
            
            cmd = [
                "uv", "run", "python", "-m", "demucs.separate",
                "-n", "htdemucs",  # 使用高质量htdemucs模型
                "--mp3",  # 输出MP3格式
                "--mp3-bitrate", "320",  # 高质量
                "-o", output_dir,
                audio_path
            ]
            
            # 对于32秒音频，增加超时时间到600秒(10分钟)
            self.logger.log("INFO", f"执行Demucs命令: {' '.join(cmd)}")
            self.logger.log("INFO", f"使用模型目录: {env['TORCH_HOME']}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
            
            if result.returncode != 0:
                return {"success": False, "error": f"Demucs 分离失败: {result.stderr}"}
            
            # 查找分离后的文件
            audio_name = Path(audio_path).stem
            demucs_subdir = os.path.join(output_dir, "htdemucs", audio_name)
            
            vocals_path = os.path.join(demucs_subdir, "vocals.mp3")
            background_paths = [
                os.path.join(demucs_subdir, "drums.mp3"),
                os.path.join(demucs_subdir, "bass.mp3"), 
                os.path.join(demucs_subdir, "other.mp3")
            ]
            
            # 验证文件存在
            if not os.path.exists(vocals_path):
                return {"success": False, "error": "Demucs 人声分离文件未生成"}
            
            # 合并非人声部分作为背景音
            background_path = os.path.join(output_dir, f"{audio_name}_background.wav")
            background_success = self._merge_background_tracks(background_paths, background_path)
            
            if not background_success:
                self.logger.log("WARNING", "背景音轨合并失败，将使用空背景音")
                background_path = None
            
            # 转换人声为WAV格式用于后续处理
            vocals_wav_path = os.path.join(output_dir, f"{audio_name}_vocals.wav")
            self._convert_to_wav(vocals_path, vocals_wav_path)
            
            self.logger.log("INFO", f"✅ Demucs 分离完成")
            self.logger.log("INFO", f"   人声: {vocals_wav_path}")
            self.logger.log("INFO", f"   背景: {background_path}")
            
            return {
                "success": True,
                "vocals_path": vocals_wav_path,
                "background_path": background_path
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Demucs 处理超时"}
        except Exception as e:
            return {"success": False, "error": f"Demucs 处理异常: {str(e)}"}
    
    def _merge_background_tracks(self, track_paths: List[str], output_path: str) -> bool:
        """合并背景音轨（drums + bass + other）"""
        try:
            existing_tracks = [p for p in track_paths if os.path.exists(p)]
            self.logger.log("INFO", f"找到{len(existing_tracks)}个背景音轨文件")
            
            if not existing_tracks:
                self.logger.log("WARNING", "没有找到背景音轨文件")
                return False
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 使用 ffmpeg 混合所有背景音轨
            if len(existing_tracks) == 1:
                # 只有一个轨道，直接转换格式
                cmd = ["ffmpeg", "-i", existing_tracks[0], "-ac", "2", "-ar", "44100", "-y", output_path]
                self.logger.log("INFO", f"单轨道转换: {existing_tracks[0]}")
            else:
                # 多个轨道，需要混合
                input_args = []
                for track in existing_tracks:
                    input_args.extend(["-i", track])
                    self.logger.log("INFO", f"添加背景音轨: {track}")
                
                # 修复filter_complex语法
                filter_inputs = "".join([f"[{i}:a]" for i in range(len(existing_tracks))])
                filter_complex = f"{filter_inputs}amix=inputs={len(existing_tracks)}:duration=longest"
                
                cmd = ["ffmpeg"] + input_args + [
                    "-filter_complex", filter_complex,
                    "-ac", "2", "-ar", "44100",
                    "-y", output_path
                ]
            
            self.logger.log("INFO", f"执行ffmpeg命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    self.logger.log("INFO", f"背景音轨合并成功: {output_path} ({file_size} bytes)")
                    return True
                else:
                    self.logger.log("ERROR", "ffmpeg成功但背景音文件未生成")
                    return False
            else:
                self.logger.log("ERROR", f"ffmpeg失败 (返回码: {result.returncode})")
                self.logger.log("ERROR", f"stderr: {result.stderr}")
                self.logger.log("ERROR", f"stdout: {result.stdout}")
                return False
            
        except subprocess.TimeoutExpired:
            self.logger.log("ERROR", "背景音轨合并超时")
            return False
        except Exception as e:
            self.logger.log("ERROR", f"背景音轨合并异常: {str(e)}")
            return False
    
    def _convert_to_wav(self, input_path: str, output_path: str):
        """转换音频为WAV格式"""
        try:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception as e:
            self.logger.log("WARNING", f"音频格式转换失败: {str(e)}")
    
    def _analyze_speakers(self, vocals_path: str) -> List[Dict]:
        """使用 pyannote.audio 分析说话人"""
        try:
            if not self.diarization_pipeline:
                return []
            
            diarization = self.diarization_pipeline(vocals_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                })
            
            self.logger.log("INFO", f"检测到 {len(set(seg['speaker'] for seg in speaker_segments))} 个说话人")
            return speaker_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"说话人分析失败: {str(e)}")
            return []
    
    def _transcribe_with_timestamps(self, vocals_path: str, language: str) -> Dict:
        """使用 whisper-timestamped 进行字级别转录"""
        try:
            if not self.whisper_model:
                return {}
            
            # 映射语言代码
            lang_map = {
                "中文": "zh", "英语": "en", "日语": "ja", "韩语": "ko",
                "法语": "fr", "德语": "de", "西班牙语": "es"
            }
            whisper_lang = lang_map.get(language, "zh")
            
            result = whisper.transcribe(
                self.whisper_model,
                vocals_path,
                language=whisper_lang,
                vad=True,  # 启用语音活动检测，减少幻觉
                compute_word_confidence=True,  # 计算词汇置信度
                refine_whisper_precision=0.5,  # 优化时间戳精度到0.5秒
                min_word_duration=0.02,  # 最小词汇持续时间20ms
                remove_empty_words=True,  # 移除可能的幻觉空词
                detect_disfluencies=False,  # 暂时关闭不流畅检测
                trust_whisper_timestamps=True  # 信任Whisper的时间戳作为基础
            )
            
            self.logger.log("INFO", f"Whisper 识别完成: {len(result.get('segments', []))} 个段落")
            return result
            
        except Exception as e:
            self.logger.log("ERROR", f"Whisper 转录失败: {str(e)}")
            return {}
    
    def _detect_multi_speaker_segments(self, semantic_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        """为每个语义片段检测说话人信息"""
        try:
            enhanced_segments = []
            
            self.logger.log("DEBUG", f"语义片段数量: {len(semantic_segments)}")
            self.logger.log("DEBUG", f"说话人片段数量: {len(speaker_segments)}")
            
            for i, segment in enumerate(semantic_segments):
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # 分析此片段内的说话人分布
                speakers_in_segment = self._analyze_speakers_in_segment(
                    segment_start, segment_end, speaker_segments
                )
                
                # 增强片段信息
                enhanced_segment = segment.copy()
                enhanced_segment.update({
                    "speakers": speakers_in_segment["speakers"],
                    "primary_speaker": speakers_in_segment["primary_speaker"],
                    "speaker_count": len(speakers_in_segment["speakers"]),
                    "multi_speaker": len(speakers_in_segment["speakers"]) > 1,
                    "speaker_confidence": speakers_in_segment["confidence"],
                    "segment_id": i + 1
                })
                
                enhanced_segments.append(enhanced_segment)
                
                # 记录多说话人片段
                if enhanced_segment["multi_speaker"]:
                    self.logger.log("INFO", f"🔍 片段{i+1}检测到多说话人: {enhanced_segment['speakers']}")
            
            # 统计信息
            multi_speaker_count = sum(1 for seg in enhanced_segments if seg["multi_speaker"])
            self.logger.log("INFO", f"🔍 多说话人检测完成: {multi_speaker_count}/{len(enhanced_segments)} 个片段包含多说话人")
            
            return enhanced_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"多说话人检测失败: {str(e)}")
            return semantic_segments  # 返回原始片段作为后备
    
    def _analyze_speakers_in_segment(self, segment_start: float, segment_end: float, 
                                   speaker_segments: List[Dict]) -> Dict:
        """分析特定时间段内的说话人分布"""
        speakers_info = {}
        
        for speaker_seg in speaker_segments:
            speaker = speaker_seg["speaker"]
            spk_start = speaker_seg["start"]
            spk_end = speaker_seg["end"]
            
            # 计算重叠时间
            overlap_start = max(segment_start, spk_start)
            overlap_end = min(segment_end, spk_end)
            
            if overlap_start < overlap_end:  # 有重叠
                overlap_duration = overlap_end - overlap_start
                
                if speaker not in speakers_info:
                    speakers_info[speaker] = {
                        "total_duration": 0,
                        "segments": []
                    }
                
                speakers_info[speaker]["total_duration"] += overlap_duration
                speakers_info[speaker]["segments"].append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "duration": overlap_duration
                })
        
        # 计算主要说话人和置信度
        if speakers_info:
            total_duration = segment_end - segment_start
            primary_speaker = max(speakers_info.keys(), 
                                key=lambda s: speakers_info[s]["total_duration"])
            
            primary_duration = speakers_info[primary_speaker]["total_duration"]
            confidence = primary_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                "speakers": list(speakers_info.keys()),
                "primary_speaker": primary_speaker,
                "confidence": confidence,
                "speaker_durations": {s: info["total_duration"] for s, info in speakers_info.items()}
            }
        else:
            return {
                "speakers": ["SPEAKER_UNKNOWN"],
                "primary_speaker": "SPEAKER_UNKNOWN", 
                "confidence": 0.0,
                "speaker_durations": {"SPEAKER_UNKNOWN": segment_end - segment_start}
            }
    
    def _align_speakers_with_words(self, speaker_segments: List[Dict], word_result: Dict) -> List[Dict]:
        """将说话人信息与文字时间戳对齐 (已废弃，由多说话人检测替代)"""
        try:
            aligned_segments = []
            
            # 调试信息
            self.logger.log("DEBUG", f"说话人片段数量: {len(speaker_segments)}")
            self.logger.log("DEBUG", f"Whisper结果包含segments: {'segments' in word_result if word_result else False}")
            
            if not word_result or "segments" not in word_result:
                self.logger.log("WARNING", "Whisper结果为空或没有segments字段")
                return []
            
            # 统计词汇数量
            total_words = 0
            for segment in word_result["segments"]:
                if "words" in segment:
                    total_words += len(segment["words"])
            
            self.logger.log("DEBUG", f"Whisper识别出总词汇数: {total_words}")
            
            for i, segment in enumerate(word_result["segments"]):
                if "words" not in segment:
                    continue
                
                for word_info in segment["words"]:
                    word_start = word_info.get("start", 0)
                    word_end = word_info.get("end", 0)
                    word_text = word_info.get("text", "").strip()  # 修复：使用'text'而不是'word'
                    
                    if not word_text:
                        continue
                    
                    # 找到对应的说话人
                    speaker = self._find_speaker_at_time(speaker_segments, word_start, word_end)
                    
                    aligned_segments.append({
                        "start": word_start,
                        "end": word_end,
                        "text": word_text,
                        "speaker": speaker
                    })
            
            self.logger.log("DEBUG", f"对齐后词汇数量: {len(aligned_segments)}")
            
            # 将连续的相同说话人的词组合成句子
            grouped_segments = self._group_consecutive_words(aligned_segments)
            
            self.logger.log("INFO", f"对齐完成: {len(grouped_segments)} 个说话人片段")
            return grouped_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"说话人文字对齐失败: {str(e)}")
            return []
    
    def _find_speaker_at_time(self, speaker_segments: List[Dict], start_time: float, end_time: float) -> str:
        """根据时间找到对应的说话人"""
        word_center = (start_time + end_time) / 2
        
        for segment in speaker_segments:
            if segment["start"] <= word_center <= segment["end"]:
                return segment["speaker"]
        
        # 如果没有完全匹配，找最近的
        closest_speaker = "SPEAKER_UNKNOWN"
        min_distance = float('inf')
        
        for segment in speaker_segments:
            seg_center = (segment["start"] + segment["end"]) / 2
            distance = abs(word_center - seg_center)
            if distance < min_distance:
                min_distance = distance
                closest_speaker = segment["speaker"]
        
        return closest_speaker
    
    def _create_semantic_segments(self, word_result: Dict) -> List[Dict]:
        """基于Whisper段落边界的智能语义切分"""
        try:
            if not word_result or "segments" not in word_result:
                self.logger.log("WARNING", "Whisper结果为空或没有segments字段")
                return []
            
            whisper_segments = word_result["segments"]
            self.logger.log("INFO", f"📝 Whisper提供了{len(whisper_segments)}个原始段落")
            
            semantic_segments = []
            
            for i, segment in enumerate(whisper_segments):
                # 提取基本信息
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                if not text or end_time <= start_time:
                    continue
                
                # 计算段落统计信息
                word_count = len(segment.get("words", []))
                duration = end_time - start_time
                
                # 计算置信度
                avg_confidence = 0.0
                if "words" in segment and segment["words"]:
                    confidences = [w.get("confidence", 0.0) for w in segment["words"] if "confidence" in w]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # 检查是否需要进一步切分（仅对过长的段落）
                if self._should_split_long_segment(segment, duration):
                    # 对过长段落进行词级切分
                    sub_segments = self._split_long_segment(segment)
                    semantic_segments.extend(sub_segments)
                    self.logger.log("INFO", f"🔪 段落{i+1}过长({duration:.2f}s)，切分为{len(sub_segments)}个子段落")
                else:
                    # 保持原始段落
                    semantic_segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "word_count": word_count,
                        "avg_confidence": avg_confidence,
                        "duration": duration,
                        "whisper_segment_id": i + 1
                    })
            
            self.logger.log("INFO", f"✂️ 智能语义切分完成: {len(whisper_segments)}个原始段落 → {len(semantic_segments)}个语义片段")
            
            # 记录切分统计信息
            if semantic_segments:
                avg_duration = sum(seg["duration"] for seg in semantic_segments) / len(semantic_segments)
                max_duration = max(seg["duration"] for seg in semantic_segments)
                min_duration = min(seg["duration"] for seg in semantic_segments)
                self.logger.log("DEBUG", f"片段时长统计: 平均{avg_duration:.2f}s, 最长{max_duration:.2f}s, 最短{min_duration:.2f}s")
            
            return semantic_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"语义切分失败: {str(e)}")
            return []
    
    def _should_split_long_segment(self, segment: Dict, duration: float) -> bool:
        """判断段落是否过长需要切分"""
        # 只切分明显过长的段落
        if duration > 15.0:  # 超过15秒必须切分
            return True
        
        # 10-15秒的段落，如果词汇数量过多也切分
        if duration > 10.0:
            word_count = len(segment.get("words", []))
            if word_count > 20:  # 超过20个词
                return True
        
        return False
    
    def _split_long_segment(self, segment: Dict) -> List[Dict]:
        """切分过长的段落"""
        if "words" not in segment or not segment["words"]:
            # 没有词级信息，无法切分，返回原段落
            return [{
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", "").strip(),
                "word_count": 0,
                "avg_confidence": 0.0,
                "duration": segment.get("end", 0) - segment.get("start", 0)
            }]
        
        words = segment["words"]
        sub_segments = []
        current_segment = {
            "start": words[0].get("start", 0),
            "end": words[0].get("end", 0),
            "text": words[0].get("text", "").strip(),
            "word_count": 1,
            "total_confidence": words[0].get("confidence", 0.0)
        }
        
        for i in range(1, len(words)):
            word = words[i]
            prev_word = words[i-1]
            
            # 计算停顿时间
            pause_duration = word.get("start", 0) - prev_word.get("end", 0)
            current_duration = current_segment["end"] - current_segment["start"]
            
            # 切分条件：长停顿或段落已经很长
            should_split = (
                pause_duration > 2.0 or  # 长停顿（2秒）
                (pause_duration > 1.0 and current_duration > 8.0) or  # 中等停顿+长段落
                current_duration > 12.0  # 强制切分超长段落
            )
            
            if should_split:
                # 完成当前段落
                current_segment["duration"] = current_segment["end"] - current_segment["start"]
                current_segment["avg_confidence"] = current_segment["total_confidence"] / current_segment["word_count"]
                del current_segment["total_confidence"]
                sub_segments.append(current_segment)
                
                # 开始新段落
                current_segment = {
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                    "text": word.get("text", "").strip(),
                    "word_count": 1,
                    "total_confidence": word.get("confidence", 0.0)
                }
            else:
                # 合并到当前段落
                current_segment["end"] = word.get("end", 0)
                current_segment["text"] += word.get("text", "").strip()
                current_segment["word_count"] += 1
                current_segment["total_confidence"] += word.get("confidence", 0.0)
        
        # 添加最后一个段落
        if current_segment:
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
            current_segment["avg_confidence"] = current_segment["total_confidence"] / current_segment["word_count"]
            del current_segment["total_confidence"]
            sub_segments.append(current_segment)
        
        return sub_segments
    
    
    def _group_consecutive_words(self, word_segments: List[Dict]) -> List[Dict]:
        """将连续的相同说话人的词组合成句子 (已废弃，由语义切分替代)"""
        if not word_segments:
            return []
        
        grouped = []
        current_group = {
            "start": word_segments[0]["start"],
            "end": word_segments[0]["end"],
            "text": word_segments[0]["text"],
            "speaker": word_segments[0]["speaker"]
        }
        
        for i in range(1, len(word_segments)):
            word = word_segments[i]
            
            # 如果是相同说话人且时间连续（间隔<2秒），则合并
            if (word["speaker"] == current_group["speaker"] and 
                word["start"] - current_group["end"] < 2.0):
                
                current_group["end"] = word["end"]
                current_group["text"] += word["text"]
            else:
                # 保存当前组，开始新组
                grouped.append(current_group.copy())
                current_group = {
                    "start": word["start"],
                    "end": word["end"], 
                    "text": word["text"],
                    "speaker": word["speaker"]
                }
        
        # 添加最后一组
        grouped.append(current_group)
        
        return grouped
    
    def _generate_audio_segments(self, vocals_path: str, enhanced_segments: List[Dict]) -> List[Dict]:
        """生成最终的音频片段（兼容新的语义切分数据结构）"""
        try:
            final_segments = []
            
            # 加载人声音频
            y, sr = librosa.load(vocals_path, sr=16000)
            
            for i, segment in enumerate(enhanced_segments):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                
                # 使用新的说话人信息
                primary_speaker = segment.get("primary_speaker", "SPEAKER_UNKNOWN")
                speakers = segment.get("speakers", [primary_speaker])
                multi_speaker = segment.get("multi_speaker", False)
                speaker_confidence = segment.get("speaker_confidence", 1.0)
                
                # 提取音频片段
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = y[start_sample:end_sample]
                
                # 生成文件名（包含多说话人标识）
                speaker_label = f"multi_{len(speakers)}" if multi_speaker else primary_speaker
                segment_path = f"./temp/professional_segment_{i+1}_{speaker_label}.wav"
                sf.write(segment_path, audio_segment, sr)
                
                # 生成增强的片段信息
                final_segment = {
                    "sequence": i + 1,
                    "timestamp": f"{start_time:.2f}-{end_time:.2f}",
                    "original_text": text,
                    "original_audio_path": segment_path,
                    "translated_text": "",
                    "translated_audio_path": "",
                    "voice_id": "",
                    "speed": 1.0,
                    
                    # 新的说话人信息
                    "primary_speaker": primary_speaker,
                    "speakers": speakers,
                    "speaker_count": len(speakers),
                    "multi_speaker": multi_speaker,
                    "speaker_confidence": speaker_confidence,
                    "speaker_durations": segment.get("speaker_durations", {}),
                    
                    # 兼容性字段
                    "speaker_id": f"speaker_{primary_speaker}",
                    
                    # 语义切分信息
                    "word_count": segment.get("word_count", 0),
                    "avg_confidence": segment.get("avg_confidence", 0.0),
                    "segment_duration": end_time - start_time
                }
                
                final_segments.append(final_segment)
                
                # 记录多说话人片段
                if multi_speaker:
                    self.logger.log("INFO", f"🎯 片段{i+1}: 多说话人 {speakers} (主要: {primary_speaker}, 置信度: {speaker_confidence:.2f})")
            
            # 统计信息
            multi_count = sum(1 for seg in final_segments if seg["multi_speaker"])
            avg_duration = sum(seg["segment_duration"] for seg in final_segments) / len(final_segments)
            self.logger.log("INFO", f"🎯 音频片段生成完成: {len(final_segments)}个片段, {multi_count}个多说话人, 平均时长{avg_duration:.2f}秒")
            
            return final_segments
            
        except Exception as e:
            self.logger.log("ERROR", f"音频片段生成失败: {str(e)}")
            return []