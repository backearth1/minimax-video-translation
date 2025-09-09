import os
import time
from typing import Dict, Any, Tuple

class AlignmentOptimizer:
    def __init__(self, config, translation_service, tts_service, logger_service):
        self.config = config
        self.translation_service = translation_service
        self.tts_service = tts_service
        self.logger = logger_service
        
    def optimize_segment(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """5步时间戳对齐优化算法"""
        segment_id = segment.get('sequence', 0)
        original_text = segment.get('original_text', '')
        translated_text = segment.get('translated_text', '')
        voice_id = segment.get('voice_id', '')
        
        self.logger.log("INFO", f"开始第{segment_id}句5步时间戳对齐优化...")
        
        # 第一步：静音裁剪检查
        step1_result = self._step1_silence_trimming(segment, target_duration)
        if step1_result["success"]:
            return step1_result
        
        # 第二步：文本优化
        step2_result = self._step2_text_optimization(segment, target_duration)
        if step2_result["success"]:
            return step2_result
        
        # 第三步：首次速度调整
        step3_result = self._step3_speed_adjustment(segment, target_duration)
        if step3_result["success"]:
            return step3_result
        
        # 第四步：速度递增重试
        step4_result = self._step4_speed_retry(segment, target_duration)
        if step4_result["success"]:
            return step4_result
        
        # 第五步：失败处理（静音）
        step5_result = self._step5_failure_handling(segment, target_duration)
        return step5_result
    
    def _step1_silence_trimming(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第一步：静音裁剪检查"""
        segment_id = segment.get('sequence', 0)
        
        self.logger.log_alignment_step(
            segment_id, 1, "静音裁剪检查", "开始处理"
        )
        
        # 首先生成初始TTS音频
        tts_result = self.tts_service.synthesize_speech(
            segment['translated_text'],
            segment['voice_id'],
            1.0,
            f"./temp/segment_{segment_id}_step1.mp3"
        )
        
        if not tts_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 1, "静音裁剪检查", f"TTS生成失败: {tts_result['error']}"
            )
            return {"success": False, "error": tts_result['error'], "step": 1}
        
        # 去除静音并获取实际时长
        trim_result = self.tts_service.trim_silence(
            tts_result["audio_path"],
            f"./temp/segment_{segment_id}_step1_trimmed.mp3"
        )
        
        if trim_result["success"]:
            actual_duration = trim_result["trimmed_duration"]
            
            if actual_duration <= target_duration:
                self.logger.log_alignment_step(
                    segment_id, 1, "静音裁剪检查", 
                    f"成功! 实际时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s",
                    {"actual_duration": actual_duration, "target_duration": target_duration}
                )
                
                return {
                    "success": True,
                    "step": 1,
                    "audio_path": trim_result["output_path"],
                    "duration": actual_duration,
                    "speed": 1.0,
                    "optimized_text": segment['translated_text'],
                    "trace_id": tts_result.get("trace_id")
                }
            else:
                self.logger.log_alignment_step(
                    segment_id, 1, "静音裁剪检查",
                    f"时长超限: 实际时长{actual_duration:.2f}s > 目标时长{target_duration:.2f}s",
                    {"actual_duration": actual_duration, "target_duration": target_duration}
                )
        else:
            # 如果静音裁剪失败，使用原始音频时长
            actual_duration = self.tts_service.get_audio_duration(tts_result["audio_path"])
            
            if actual_duration <= target_duration:
                self.logger.log_alignment_step(
                    segment_id, 1, "静音裁剪检查",
                    f"成功! 原始时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s"
                )
                
                return {
                    "success": True,
                    "step": 1,
                    "audio_path": tts_result["audio_path"],
                    "duration": actual_duration,
                    "speed": 1.0,
                    "optimized_text": segment['translated_text'],
                    "trace_id": tts_result.get("trace_id")
                }
        
        return {"success": False, "step": 1, "actual_duration": getattr(self, '_last_duration', 3.0)}
    
    def _step2_text_optimization(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第二步：文本优化"""
        segment_id = segment.get('sequence', 0)
        
        self.logger.log_alignment_step(
            segment_id, 2, "文本优化", "开始优化翻译文本"
        )
        
        # 估算当前时长（如果没有实际音频）
        estimated_duration = len(segment['translated_text']) * 0.15  # 粗略估算
        
        # 调用翻译优化服务
        optimize_result = self.translation_service.optimize_translation(
            segment.get('original_text', ''),
            segment['translated_text'],
            estimated_duration,
            target_duration,
            self.config.target_language
        )
        
        if not optimize_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化", f"优化失败: {optimize_result['error']}"
            )
            return {"success": False, "error": optimize_result['error'], "step": 2}
        
        optimized_text = optimize_result["optimized_text"]
        
        # 用优化后的文本重新生成TTS
        tts_result = self.tts_service.synthesize_speech(
            optimized_text,
            segment['voice_id'],
            1.0,
            f"./temp/segment_{segment_id}_step2.mp3"
        )
        
        if not tts_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化", f"优化后TTS生成失败: {tts_result['error']}"
            )
            return {"success": False, "error": tts_result['error'], "step": 2}
        
        # 检查优化后的时长
        actual_duration = self.tts_service.get_audio_duration(tts_result["audio_path"])
        
        if actual_duration <= target_duration:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化",
                f"成功! 优化后时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s",
                {
                    "original_text": segment['translated_text'],
                    "optimized_text": optimized_text,
                    "actual_duration": actual_duration
                }
            )
            
            return {
                "success": True,
                "step": 2,
                "audio_path": tts_result["audio_path"],
                "duration": actual_duration,
                "speed": 1.0,
                "optimized_text": optimized_text,
                "optimization_trace_id": optimize_result.get("trace_id"),
                "tts_trace_id": tts_result.get("trace_id")
            }
        else:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化",
                f"仍然超时: 优化后时长{actual_duration:.2f}s > 目标时长{target_duration:.2f}s"
            )
        
        # 存储信息供下一步使用
        self._last_duration = actual_duration
        self._last_optimized_text = optimized_text
        
        return {
            "success": False, 
            "step": 2, 
            "actual_duration": actual_duration,
            "optimized_text": optimized_text,
            "audio_path": tts_result["audio_path"]
        }
    
    def _step3_speed_adjustment(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第三步：首次速度调整"""
        segment_id = segment.get('sequence', 0)
        
        # 获取上一步的结果
        current_duration = getattr(self, '_last_duration', len(segment['translated_text']) * 0.15)
        optimized_text = getattr(self, '_last_optimized_text', segment['translated_text'])
        
        # 计算需要的速度
        speed = min(2.0, current_duration / target_duration + 0.2)
        
        self.logger.log_alignment_step(
            segment_id, 3, "首次速度调整",
            f"计算速度参数: {speed:.1f}",
            {"current_duration": current_duration, "target_duration": target_duration, "speed": speed}
        )
        
        # 使用速度参数重新生成TTS
        tts_result = self.tts_service.synthesize_speech(
            optimized_text,
            segment['voice_id'],
            speed,
            f"./temp/segment_{segment_id}_step3.mp3"
        )
        
        if not tts_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 3, "首次速度调整", f"加速TTS生成失败: {tts_result['error']}"
            )
            return {"success": False, "error": tts_result['error'], "step": 3}
        
        actual_duration = self.tts_service.get_audio_duration(tts_result["audio_path"])
        
        if actual_duration <= target_duration:
            self.logger.log_alignment_step(
                segment_id, 3, "首次速度调整",
                f"成功! 加速后时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s"
            )
            
            return {
                "success": True,
                "step": 3,
                "audio_path": tts_result["audio_path"],
                "duration": actual_duration,
                "speed": speed,
                "optimized_text": optimized_text,
                "trace_id": tts_result.get("trace_id")
            }
        else:
            self.logger.log_alignment_step(
                segment_id, 3, "首次速度调整",
                f"仍然超时: 加速后时长{actual_duration:.2f}s > 目标时长{target_duration:.2f}s"
            )
        
        # 存储信息供下一步使用
        self._last_speed = speed
        
        return {"success": False, "step": 3, "actual_duration": actual_duration, "speed": speed}
    
    def _step4_speed_retry(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第四步：速度递增重试"""
        segment_id = segment.get('sequence', 0)
        
        # 获取上一步的速度
        last_speed = getattr(self, '_last_speed', 1.5)
        optimized_text = getattr(self, '_last_optimized_text', segment['translated_text'])
        
        # 递增速度重试
        speeds_to_try = [last_speed + 0.5, 2.0]
        
        for speed in speeds_to_try:
            if speed > 2.0:
                speed = 2.0
                
            self.logger.log_alignment_step(
                segment_id, 4, "速度递增重试",
                f"尝试速度参数: {speed:.1f}"
            )
            
            tts_result = self.tts_service.synthesize_speech(
                optimized_text,
                segment['voice_id'],
                speed,
                f"./temp/segment_{segment_id}_step4_{speed:.1f}.mp3"
            )
            
            if tts_result["success"]:
                actual_duration = self.tts_service.get_audio_duration(tts_result["audio_path"])
                
                if actual_duration <= target_duration:
                    self.logger.log_alignment_step(
                        segment_id, 4, "速度递增重试",
                        f"成功! 速度{speed:.1f}时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s"
                    )
                    
                    return {
                        "success": True,
                        "step": 4,
                        "audio_path": tts_result["audio_path"],
                        "duration": actual_duration,
                        "speed": speed,
                        "optimized_text": optimized_text,
                        "trace_id": tts_result.get("trace_id")
                    }
        
        self.logger.log_alignment_step(
            segment_id, 4, "速度递增重试", "所有速度尝试均失败"
        )
        
        return {"success": False, "step": 4}
    
    def _step5_failure_handling(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第五步：失败处理（设为静音）"""
        segment_id = segment.get('sequence', 0)
        
        self.logger.log_alignment_step(
            segment_id, 5, "失败处理",
            f"所有优化步骤失败，设置为静音（时长: {target_duration:.2f}s）"
        )
        
        # 生成静音音频文件
        silence_path = f"./temp/segment_{segment_id}_silence.mp3"
        silence_result = self._generate_silence_audio(target_duration, silence_path)
        
        if silence_result["success"]:
            return {
                "success": True,
                "step": 5,
                "audio_path": silence_path,
                "duration": target_duration,
                "speed": 1.0,
                "optimized_text": "",
                "is_silence": True
            }
        else:
            return {
                "success": False,
                "step": 5,
                "error": "静音音频生成失败"
            }
    
    def _generate_silence_audio(self, duration: float, output_path: str) -> Dict[str, Any]:
        """生成指定时长的静音音频"""
        try:
            import numpy as np
            import soundfile as sf
            
            # 生成静音数据
            sample_rate = 44100
            samples = int(duration * sample_rate)
            silence = np.zeros(samples, dtype=np.float32)
            
            # 保存为音频文件
            sf.write(output_path, silence, sample_rate)
            
            return {"success": True, "file_path": output_path}
            
        except ImportError:
            # 如果没有soundfile，使用ffmpeg
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=duration={duration}',
                    '-c:a', 'mp3', '-y', output_path
                ]
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0:
                    return {"success": True, "file_path": output_path}
                else:
                    return {"success": False, "error": "ffmpeg生成静音失败"}
                    
            except Exception as e:
                return {"success": False, "error": f"静音生成异常: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"静音生成异常: {str(e)}"}
    
    def batch_optimize(self, segments: list) -> list:
        """批量优化多个片段"""
        results = []
        
        for i, segment in enumerate(segments):
            self.logger.log("INFO", f"正在优化第{i+1}/{len(segments)}个片段...")
            
            # 解析时间戳获取目标时长
            timestamp = segment.get('timestamp', '0-3')
            start_time, end_time = self._parse_timestamp(timestamp)
            target_duration = end_time - start_time
            
            result = self.optimize_segment(segment, target_duration)
            result['segment_id'] = segment.get('sequence', i)
            results.append(result)
            
            # 批量处理时添加小延迟
            time.sleep(0.5)
                
        return results
    
    def _parse_timestamp(self, timestamp: str) -> Tuple[float, float]:
        """解析时间戳"""
        try:
            parts = timestamp.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                return start, end
        except:
            pass
        return 0.0, 3.0  # 默认值