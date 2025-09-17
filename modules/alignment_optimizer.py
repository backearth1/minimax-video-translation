import os
import time
from typing import Dict, Any, Tuple

class AlignmentOptimizer:
    def __init__(self, config, translation_service, tts_service, logger_service):
        self.config = config
        self.translation_service = translation_service
        self.tts_service = tts_service
        self.logger = logger_service
        
    def _get_trimmed_duration(self, audio_path: str, segment_id: int, step: str) -> float:
        """获取去除静音后的音频时长"""
        # 先尝试静音裁剪
        trimmed_path = f"./temp/segment_{segment_id}_{step}_trimmed.mp3"
        trim_result = self.tts_service.trim_silence(audio_path, trimmed_path)
        
        if trim_result["success"]:
            trimmed_duration = trim_result["trimmed_duration"]
            self.logger.log("DEBUG", f"音频静音裁剪成功: {audio_path} -> {trimmed_duration:.2f}s")
            return trimmed_duration
        else:
            # 如果裁剪失败，使用原始时长
            original_duration = self.tts_service.get_audio_duration(audio_path)
            self.logger.log("WARNING", f"音频静音裁剪失败，使用原始时长: {original_duration:.2f}s")
            return original_duration

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
        """第一步：状态分析和静音裁剪检查"""
        segment_id = segment.get('sequence', 0)
        
        # 检查是否已有翻译音频文件
        existing_audio_path = segment.get('translated_audio_path')
        current_duration = 0.0
        ratio = 0.0
        
        if existing_audio_path and os.path.exists(existing_audio_path):
            # 获取现有音频时长
            current_duration = self.tts_service.get_audio_duration(existing_audio_path)
            ratio = current_duration / target_duration if target_duration > 0 else 999.0
            
            self.logger.log_alignment_step(
                segment_id, 1, "状态分析", 
                f"当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
                f"当前比例: {ratio:.2f}, 比例<1: {ratio < 1.0}, 下一步措施: {'直接使用' if ratio <= 1.0 else '需要优化'}"
            )
            
            # 如果比例小于等于1，说明时长合适，直接使用
            if ratio <= 1.0:
                self.logger.log_alignment_step(
                    segment_id, 1, "状态分析", 
                    f"成功! 现有音频时长{current_duration:.2f}s ≤ 目标时长{target_duration:.2f}s，直接使用",
                    {"current_duration": current_duration, "target_duration": target_duration}
                )
                
                return {
                    "success": True,
                    "step": 1,
                    "audio_path": existing_audio_path,
                    "duration": current_duration,
                    "speed": 1.0,
                    "optimized_text": segment['translated_text'],
                    "ratio": round(ratio, 2)
                }
        else:
            # 没有现有音频，需要先生成
            estimated_duration = len(segment.get('translated_text', '')) * 0.15  # 粗略估算
            ratio = estimated_duration / target_duration if target_duration > 0 else 999.0
            
            self.logger.log_alignment_step(
                segment_id, 1, "状态分析", 
                f"当前声音时长: 无现有音频(估算{estimated_duration:.2f}s), 目标时长: {target_duration:.2f}s, "
                f"当前比例: {ratio:.2f}, 比例<1: {ratio < 1.0}, 下一步措施: 生成TTS音频"
            )
            
            # 生成初始TTS音频
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
            
            current_duration = self._get_trimmed_duration(tts_result["audio_path"], segment_id, "step1")
            ratio = current_duration / target_duration if target_duration > 0 else 999.0
            
            self.logger.log_alignment_step(
                segment_id, 1, "状态分析", 
                f"TTS生成后 - 当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
                f"当前比例: {ratio:.2f}, 比例<1: {ratio < 1.0}, 下一步措施: {'静音裁剪检查' if ratio > 1.0 else '直接使用'}"
            )
            
            if ratio <= 1.0:
                self.logger.log_alignment_step(
                    segment_id, 1, "静音裁剪检查", 
                    f"成功! 生成音频时长{current_duration:.2f}s ≤ 目标时长{target_duration:.2f}s",
                    {"current_duration": current_duration, "target_duration": target_duration}
                )
                
                return {
                    "success": True,
                    "step": 1,
                    "audio_path": tts_result["audio_path"],
                    "duration": current_duration,
                    "speed": 1.0,
                    "optimized_text": segment['translated_text'],
                    "trace_id": tts_result.get("trace_id"),
                    "ratio": round(ratio, 2)
                }
            
            existing_audio_path = tts_result["audio_path"]
        
        # 如果到这里，说明音频时长超过目标，尝试静音裁剪
        self.logger.log_alignment_step(
            segment_id, 1, "静音裁剪检查", 
            f"音频超时，尝试静音裁剪: {current_duration:.2f}s > {target_duration:.2f}s"
        )
        
        trim_result = self.tts_service.trim_silence(
            existing_audio_path,
            f"./temp/segment_{segment_id}_step1_trimmed.mp3"
        )
        
        if trim_result["success"]:
            trimmed_duration = trim_result["trimmed_duration"]
            new_ratio = trimmed_duration / target_duration if target_duration > 0 else 999.0
            
            self.logger.log_alignment_step(
                segment_id, 1, "静音裁剪检查", 
                f"裁剪后 - 当前声音时长: {trimmed_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
                f"当前比例: {new_ratio:.2f}, 比例<1: {new_ratio < 1.0}, 下一步措施: {'成功' if new_ratio <= 1.0 else '需要文本优化'}"
            )
            
            if new_ratio <= 1.0:
                self.logger.log_alignment_step(
                    segment_id, 1, "静音裁剪检查", 
                    f"成功! 裁剪后时长{trimmed_duration:.2f}s ≤ 目标时长{target_duration:.2f}s",
                    {"trimmed_duration": trimmed_duration, "target_duration": target_duration}
                )
                
                return {
                    "success": True,
                    "step": 1,
                    "audio_path": trim_result["output_path"],
                    "duration": trimmed_duration,
                    "speed": 1.0,
                    "optimized_text": segment['translated_text'],
                    "ratio": round(new_ratio, 2)
                }
            else:
                current_duration = trimmed_duration
        
        # 记录失败信息供下一步使用
        self._last_duration = current_duration
        self.logger.log_alignment_step(
            segment_id, 1, "静音裁剪检查",
            f"失败: 裁剪后时长{current_duration:.2f}s > 目标时长{target_duration:.2f}s, 进入文本优化"
        )
        
        return {"success": False, "step": 1, "actual_duration": current_duration}
    
    def _step2_text_optimization(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第二步：文本优化"""
        segment_id = segment.get('sequence', 0)
        
        # 获取当前状态
        current_duration = getattr(self, '_last_duration', 0.0)
        current_ratio = current_duration / target_duration if target_duration > 0 else 999.0
        original_text = segment['translated_text']
        original_char_count = len(original_text)
        
        # 计算目标字符数（基于比例）
        target_char_count = int(original_char_count / current_ratio) if current_ratio > 1.0 else original_char_count
        
        self.logger.log_alignment_step(
            segment_id, 2, "文本优化", 
            f"当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {current_ratio:.2f}, 比例<1: {current_ratio < 1.0}, "
            f"下一步措施: 调用LLM优化文本 - 原文本字符数: {original_char_count}, 目标字符数: {target_char_count}"
        )
        
        # 调用翻译优化服务
        optimize_result = self.translation_service.optimize_translation(
            segment.get('original_text', ''),
            original_text,
            current_duration,
            target_duration,
            self.config.target_language
        )
        
        if not optimize_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化", f"LLM优化失败: {optimize_result['error']}"
            )
            return {"success": False, "error": optimize_result['error'], "step": 2}
        
        optimized_text = optimize_result["optimized_text"]
        final_char_count = len(optimized_text)
        optimization_trace_id = optimize_result.get("trace_id", "")
        
        # 打印详细的优化结果
        self.logger.log_alignment_step(
            segment_id, 2, "文本优化", 
            f"LLM优化完成 - Trace-ID: {optimization_trace_id}, "
            f"原文本字符数: {original_char_count}, 目标字符数: {target_char_count}, "
            f"最终返回字符数: {final_char_count}, 压缩比例: {final_char_count/original_char_count:.2f}"
        )
        
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
        
        # 检查优化后的时长（去除静音）
        actual_duration = self._get_trimmed_duration(tts_result["audio_path"], segment_id, "step2")
        new_ratio = actual_duration / target_duration if target_duration > 0 else 999.0
        
        self.logger.log_alignment_step(
            segment_id, 2, "文本优化", 
            f"TTS生成后 - 当前声音时长: {actual_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {new_ratio:.2f}, 比例<1: {new_ratio < 1.0}, "
            f"下一步措施: {'成功' if new_ratio <= 1.0 else '需要速度调整'}"
        )
        
        if new_ratio <= 1.0:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化",
                f"成功! 优化后时长{actual_duration:.2f}s ≤ 目标时长{target_duration:.2f}s",
                {
                    "original_text": original_text,
                    "optimized_text": optimized_text,
                    "actual_duration": actual_duration,
                    "char_reduction": f"{original_char_count} → {final_char_count}"
                }
            )
            
            return {
                "success": True,
                "step": 2,
                "audio_path": tts_result["audio_path"],
                "duration": actual_duration,
                "speed": 1.0,
                "optimized_text": optimized_text,
                "optimization_trace_id": optimization_trace_id,
                "tts_trace_id": tts_result.get("trace_id"),
                "ratio": round(new_ratio, 2)
            }
        else:
            self.logger.log_alignment_step(
                segment_id, 2, "文本优化",
                f"仍然超时: 优化后时长{actual_duration:.2f}s > 目标时长{target_duration:.2f}s, 进入速度调整"
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
        current_ratio = current_duration / target_duration if target_duration > 0 else 999.0
        optimized_text = getattr(self, '_last_optimized_text', segment['translated_text'])
        
        # 计算需要的速度，保留2位小数
        speed = round(min(2.0, current_duration / target_duration + 0.2), 2)
        
        self.logger.log_alignment_step(
            segment_id, 3, "首次速度调整",
            f"当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {current_ratio:.2f}, 比例<1: {current_ratio < 1.0}, "
            f"下一步措施: 速度调整到{speed:.1f}倍"
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
        
        actual_duration = self._get_trimmed_duration(tts_result["audio_path"], segment_id, "step3")
        new_ratio = actual_duration / target_duration if target_duration > 0 else 999.0
        
        self.logger.log_alignment_step(
            segment_id, 3, "首次速度调整",
            f"速度调整后 - 当前声音时长: {actual_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {new_ratio:.2f}, 比例<1: {new_ratio < 1.0}, "
            f"下一步措施: {'成功' if new_ratio <= 1.0 else '需要更高速度'}"
        )
        
        if new_ratio <= 1.0:
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
                "trace_id": tts_result.get("trace_id"),
                "ratio": round(new_ratio, 2)
            }
        else:
            self.logger.log_alignment_step(
                segment_id, 3, "首次速度调整",
                f"仍然超时: 加速后时长{actual_duration:.2f}s > 目标时长{target_duration:.2f}s, 进入递增重试"
            )
        
        # 存储信息供下一步使用
        self._last_speed = round(speed, 2)
        self._last_duration = actual_duration
        
        return {"success": False, "step": 3, "actual_duration": actual_duration, "speed": speed}
    
    def _step4_speed_retry(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第四步：速度递增重试"""
        segment_id = segment.get('sequence', 0)
        
        # 获取上一步的结果
        last_speed = getattr(self, '_last_speed', 1.5)
        current_duration = getattr(self, '_last_duration', 0.0)
        current_ratio = current_duration / target_duration if target_duration > 0 else 999.0
        optimized_text = getattr(self, '_last_optimized_text', segment['translated_text'])
        
        # 递增速度重试
        speeds_to_try = [round(last_speed + 0.5, 2), 2.0]
        
        self.logger.log_alignment_step(
            segment_id, 4, "速度递增重试",
            f"当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {current_ratio:.2f}, 比例<1: {current_ratio < 1.0}, "
            f"下一步措施: 尝试更高速度 {speeds_to_try}"
        )
        
        for speed in speeds_to_try:
            if speed > 2.0:
                speed = 2.0
            speed = round(speed, 2)
                
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
                actual_duration = self._get_trimmed_duration(tts_result["audio_path"], segment_id, f"step4_{speed:.1f}")
                new_ratio = actual_duration / target_duration if target_duration > 0 else 999.0
                
                self.logger.log_alignment_step(
                    segment_id, 4, "速度递增重试",
                    f"速度{speed:.1f}测试 - 当前声音时长: {actual_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
                    f"当前比例: {new_ratio:.2f}, 比例<1: {new_ratio < 1.0}, "
                    f"下一步措施: {'成功' if new_ratio <= 1.0 else '继续尝试更高速度'}"
                )
                
                if new_ratio <= 1.0:
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
                        "trace_id": tts_result.get("trace_id"),
                        "ratio": round(new_ratio, 2)
                    }
                else:
                    # 保存最后一次尝试的时长供step 5使用
                    self._last_duration = actual_duration
        
        # 获取最后一次尝试的实际时长（去除静音后）
        last_duration = getattr(self, '_last_duration', current_duration)
        last_ratio = last_duration / target_duration if target_duration > 0 else 999.0
        
        self.logger.log_alignment_step(
            segment_id, 4, "速度递增重试", 
            f"所有速度尝试均失败 - 当前声音时长: {last_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {last_ratio:.2f}, 比例<1: {last_ratio < 1.0}, "
            f"下一步措施: 设为静音"
        )
        
        return {"success": False, "step": 4}
    
    def _step5_failure_handling(self, segment: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """第五步：失败处理（设为静音）"""
        segment_id = segment.get('sequence', 0)
        
        current_duration = getattr(self, '_last_duration', 0.0)
        current_ratio = current_duration / target_duration if target_duration > 0 else 999.0
        
        self.logger.log_alignment_step(
            segment_id, 5, "失败处理",
            f"当前声音时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s, "
            f"当前比例: {current_ratio:.2f}, 比例<1: {current_ratio < 1.0}, "
            f"下一步措施: 所有优化步骤失败，设置为静音（时长: {target_duration:.2f}s）"
        )
        
        # 生成静音音频文件
        silence_path = f"./temp/segment_{segment_id}_silence.mp3"
        silence_result = self._generate_silence_audio(target_duration, silence_path)
        
        if silence_result["success"]:
            self.logger.log_alignment_step(
                segment_id, 5, "失败处理",
                f"静音生成成功 - 当前声音时长: 0.0s, 目标时长: {target_duration:.2f}s, "
                f"当前比例: 0.0, 比例<1: True, 下一步措施: 完成"
            )
            
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