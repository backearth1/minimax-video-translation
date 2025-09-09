import requests
import json
import time
from typing import Dict, Any, Optional

class TranslationService:
    def __init__(self, config, rate_limiter, logger_service):
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = logger_service
        
    def translate_text(self, original_text: str, target_language: str = None) -> Dict[str, Any]:
        """翻译文本"""
        if not target_language:
            target_language = self.config.target_language
            
        # 检查限流
        self.rate_limiter.wait_for_availability('llm')
        
        url = f"{self.config.api_endpoint}/v1/text/chatcompletion_v2"
        headers = self.config.get_api_headers()
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": f"你是一个专业的翻译专家，请将用户提供的文本翻译成{target_language}，保持原文的语气和情感。"
                },
                {
                    "role": "user",
                    "content": f"请将以下文本翻译成{target_language}：\n{original_text}"
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                trace_id = response_data.get('trace_id', response.headers.get('Trace-ID', ''))
                
                # 打印详细调试信息
                print(f"LLM API Response Headers: {dict(response.headers)}")
                print(f"LLM API Trace-ID: {trace_id}")
                print(f"LLM API Response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                
                if 'choices' in response_data and response_data['choices']:
                    translated_text = response_data['choices'][0]['message']['content'].strip()
                    
                    self.logger.log_api_call(
                        "LLM翻译", url, trace_id, "success", duration
                    )
                    
                    return {
                        "success": True,
                        "translated_text": translated_text,
                        "trace_id": trace_id,
                        "original_text": original_text,
                        "target_language": target_language
                    }
                else:
                    error_msg = "API响应格式错误"
                    self.logger.log("ERROR", f"翻译失败: {error_msg}", trace_id)
                    return {"success": False, "error": error_msg}
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                self.logger.log("ERROR", f"翻译失败: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"翻译请求异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def optimize_translation(self, original_text: str, current_translation: str, 
                           current_duration: float, target_duration: float,
                           target_language: str = None) -> Dict[str, Any]:
        """优化翻译文本以适应目标时长"""
        if not target_language:
            target_language = self.config.target_language
            
        current_char_count = len(current_translation)
        target_char_count = int(current_char_count * target_duration / current_duration)
        
        # 检查限流
        self.rate_limiter.wait_for_availability('llm')
        
        url = f"{self.config.api_endpoint}/v1/text/chatcompletion_v2"
        headers = self.config.get_api_headers()
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个翻译优化专家，擅长在保持原意的基础上精简文本"
                },
                {
                    "role": "user",
                    "content": f"""你的任务是翻译优化，原文"{original_text}"当前"{target_language}"翻译"{current_translation}"，你需要缩短翻译的文字，同时保持口语化表达，当前字符数是{current_char_count}个字，需要精简成少于{target_char_count}个字，翻译如下："""
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                trace_id = response_data.get('trace_id', '')
                
                if 'choices' in response_data and response_data['choices']:
                    optimized_text = response_data['choices'][0]['message']['content'].strip()
                    
                    self.logger.log_api_call(
                        "LLM文本优化", url, trace_id, "success", duration
                    )
                    
                    return {
                        "success": True,
                        "optimized_text": optimized_text,
                        "trace_id": trace_id,
                        "original_char_count": current_char_count,
                        "target_char_count": target_char_count,
                        "actual_char_count": len(optimized_text)
                    }
                else:
                    error_msg = "API响应格式错误"
                    self.logger.log("ERROR", f"文本优化失败: {error_msg}", trace_id)
                    return {"success": False, "error": error_msg}
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                self.logger.log("ERROR", f"文本优化失败: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"文本优化请求异常: {str(e)}"
            self.logger.log("ERROR", error_msg)
            return {"success": False, "error": error_msg}
    
    def batch_translate(self, text_list: list, target_language: str = None) -> list:
        """批量翻译"""
        results = []
        
        for i, text in enumerate(text_list):
            self.logger.log("INFO", f"正在翻译第{i+1}/{len(text_list)}句...")
            result = self.translate_text(text, target_language)
            results.append(result)
            
            # 批量处理时添加小延迟
            if i < len(text_list) - 1:
                time.sleep(0.5)
                
        return results