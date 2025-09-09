import os
import requests
from typing import Dict, List, Optional
from pathlib import Path

class ModelManager:
    """
    AI模型管理器
    - 统一管理所有需要的AI模型
    - 检查本地缓存，避免重复下载
    - 提供清晰的下载进度提示
    """
    
    def __init__(self, logger_service):
        self.logger = logger_service
        
        # 项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        
        # 定义项目需要的所有模型 - 使用统一的项目模型目录
        self.required_models = {
            # Whisper ASR模型 (优先级从高到低)
            "whisper": {
                "models": ["base", "medium", "large-v2"],
                "priority": "large-v2",  # 专业处理优先使用large-v2模型
                "description": "语音识别模型",
                "cache_dir": os.path.join(self.models_dir, "whisper")
            },
            
            # pyannote.audio 说话人分离模型
            "pyannote": {
                "models": ["pyannote/speaker-diarization-3.1"],
                "priority": "pyannote/speaker-diarization-3.1",
                "description": "说话人分离模型",
                "cache_dir": os.path.join(self.models_dir, "pyannote")
            },
            
            # Demucs 音频源分离模型
            "demucs": {
                "models": ["955717e8-8726e21a.th"],  # 直接使用文件名
                "priority": "955717e8-8726e21a.th", 
                "description": "音频源分离模型",
                "cache_dir": os.path.join(self.models_dir, "demucs", "checkpoints")  # 使用checkpoints子目录
            }
        }
    
    def check_model_availability(self) -> Dict[str, Dict]:
        """检查所有模型的可用性"""
        status = {}
        
        for model_type, config in self.required_models.items():
            status[model_type] = {
                "available": False,
                "cached_models": [],
                "missing_models": [],
                "priority_model": config["priority"],
                "description": config["description"]
            }
            
            if model_type == "whisper":
                status[model_type] = self._check_whisper_models(config)
            elif model_type == "pyannote":
                status[model_type] = self._check_pyannote_models(config)
            elif model_type == "demucs":
                status[model_type] = self._check_demucs_models(config)
        
        return status
    
    def _check_whisper_models(self, config: Dict) -> Dict:
        """检查Whisper模型"""
        cache_dir = Path(config["cache_dir"])
        cached_models = []
        
        if cache_dir.exists():
            # 检查已下载的模型文件
            for model_name in config["models"]:
                model_file = cache_dir / f"{model_name}.pt"
                if model_file.exists():
                    cached_models.append(model_name)
        
        missing_models = [m for m in config["models"] if m not in cached_models]
        
        return {
            "available": len(cached_models) > 0,
            "cached_models": cached_models,
            "missing_models": missing_models,
            "priority_model": config["priority"],
            "description": config["description"]
        }
    
    def _check_pyannote_models(self, config: Dict) -> Dict:
        """检查pyannote.audio模型"""
        cache_dir = Path(config["cache_dir"])
        cached_models = []
        
        if cache_dir.exists():
            # 检查已下载的模型目录
            for model_name in config["models"]:
                # pyannote模型以特殊格式存储
                model_dir_name = f"models--{model_name.replace('/', '--')}"
                model_dir = cache_dir / model_dir_name
                if model_dir.exists():
                    cached_models.append(model_name)
                    self.logger.log("DEBUG", f"找到pyannote模型: {model_dir}")
        
        missing_models = [m for m in config["models"] if m not in cached_models]
        
        return {
            "available": len(cached_models) > 0,
            "cached_models": cached_models,
            "missing_models": missing_models,
            "priority_model": config["priority"],
            "description": config["description"]
        }
    
    def _check_demucs_models(self, config: Dict) -> Dict:
        """检查Demucs模型"""
        cache_dir = Path(config["cache_dir"])
        cached_models = []
        
        if cache_dir.exists():
            # 检查已下载的模型文件
            for model_name in config["models"]:
                model_file = cache_dir / model_name
                if model_file.exists():
                    cached_models.append(model_name)
        
        missing_models = [m for m in config["models"] if m not in cached_models]
        
        return {
            "available": len(cached_models) > 0,
            "cached_models": cached_models,
            "missing_models": missing_models,
            "priority_model": config["priority"],
            "description": config["description"]
        }
    
    def get_recommended_model(self, model_type: str, available_memory_gb: float = 4.0, prioritize_quality: bool = True) -> Optional[str]:
        """根据质量或系统资源推荐最佳模型"""
        if model_type not in self.required_models:
            return None
        
        config = self.required_models[model_type]
        
        if model_type == "whisper":
            if prioritize_quality:
                # 专业AI翻译：优先使用最高质量模型
                if "large-v2" in config["models"]:
                    return "large-v2"  # 最高质量
                elif "medium" in config["models"]:
                    return "medium"    # 中等质量
                else:
                    return "base"      # 基础质量
            else:
                # 快速处理：根据内存推荐
                if available_memory_gb < 2:
                    return "base"  # 最轻量级
                elif available_memory_gb < 4:
                    return "medium" if "medium" in config["models"] else "base"
                else:
                    return "large-v2"  # 最高质量
        
        return config["priority"]
    
    def print_model_status(self):
        """打印模型状态报告"""
        self.logger.log("INFO", "📋 AI模型状态检查报告")
        self.logger.log("INFO", "=" * 50)
        
        status = self.check_model_availability()
        
        for model_type, info in status.items():
            self.logger.log("INFO", f"🔍 {info['description']} ({model_type})")
            
            if info["available"]:
                self.logger.log("INFO", f"   ✅ 已缓存: {', '.join(info['cached_models'])}")
                self.logger.log("INFO", f"   🎯 推荐使用: {info['priority_model']}")
            else:
                self.logger.log("WARNING", f"   ❌ 未找到缓存模型")
                
            if info["missing_models"]:
                self.logger.log("WARNING", f"   📥 需要下载: {', '.join(info['missing_models'])}")
            
            self.logger.log("INFO", "")
    
    def prepare_models_for_professional_processing(self) -> Dict[str, str]:
        """
        为专业音频处理准备模型
        返回推荐的模型配置（优先质量）
        """
        self.logger.log("INFO", "🚀 准备专业音频处理模型...")
        
        # 检查系统内存
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            self.logger.log("INFO", f"📊 系统可用内存: {available_gb:.1f}GB")
        except ImportError:
            available_gb = 4.0  # 默认假设4GB
            self.logger.log("WARNING", "无法检测系统内存，假设4GB可用")
        
        # 检查模型状态
        self.print_model_status()
        
        # 推荐模型配置（专业处理优先质量）
        recommended_config = {}
        
        for model_type in ["whisper", "pyannote", "demucs"]:
            # 专业处理模式：prioritize_quality=True
            recommended_model = self.get_recommended_model(model_type, available_gb, prioritize_quality=True)
            recommended_config[model_type] = recommended_model
            
            # 检查推荐模型是否可用
            status = self.check_model_availability()
            if not status[model_type]["available"]:
                self.logger.log("WARNING", f"⚠️  {model_type} 模型需要首次下载，可能需要几分钟时间")
        
        # 特别提醒Whisper模型选择
        whisper_model = recommended_config.get("whisper", "base")
        if whisper_model == "large-v2":
            self.logger.log("INFO", "🎯 专业模式：使用Whisper large-v2模型确保最佳识别效果")
        elif whisper_model == "medium":
            self.logger.log("INFO", "🎯 专业模式：使用Whisper medium模型平衡效果与速度")
        else:
            self.logger.log("WARNING", "⚠️  专业模式：使用Whisper base模型，识别效果可能受限")
        
        self.logger.log("INFO", "🎯 专业AI处理配置（优先质量）:")
        for model_type, model_name in recommended_config.items():
            self.logger.log("INFO", f"   {model_type}: {model_name}")
        
        return recommended_config
    
    def estimate_download_time(self, model_type: str) -> str:
        """估算模型下载时间"""
        download_estimates = {
            "whisper": {
                "base": "1-2分钟 (~39MB)",
                "medium": "3-5分钟 (~769MB)", 
                "large-v2": "5-10分钟 (~1.5GB)"
            },
            "pyannote": {
                "pyannote/speaker-diarization-3.1": "2-3分钟 (~100MB)"
            },
            "demucs": {
                "htdemucs": "3-5分钟 (~200MB)"
            }
        }
        
        if model_type in download_estimates:
            return download_estimates[model_type]
        
        return "未知"