import logging
from datetime import datetime
from typing import List, Dict
from collections import deque

class LoggerService:
    def __init__(self, max_logs=1000):
        self.logs = deque(maxlen=max_logs)
        self.logger = logging.getLogger(__name__)
        
        # 日志级别映射
        self.level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
            "API": logging.INFO,      # API调用日志
            "ALIGN": logging.INFO,   # 对齐处理日志
            "PROCESS": logging.INFO  # 处理过程日志
        }
    
    def log(self, level: str, message: str, trace_id: str = None, context: Dict = None):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "trace_id": trace_id,
            "context": context or {}
        }
        
        self.logs.append(log_entry)
        
        # 同时写入Python日志系统
        log_level = self.level_map.get(level, logging.INFO)
        log_message = f"[{level}] {message}"
        if trace_id:
            log_message += f" (Trace-ID: {trace_id})"
        
        self.logger.log(log_level, log_message)
    
    def log_api_call(self, api_type: str, endpoint: str, trace_id: str = None, 
                     status: str = "success", duration: float = None):
        """记录API调用"""
        message = f"{api_type} API调用: {endpoint}"
        if duration:
            message += f" (耗时: {duration:.2f}s)"
        if status != "success":
            message += f" - {status}"
            
        self.log("API", message, trace_id)
    
    def log_alignment_step(self, segment_id: int, step: int, step_name: str, 
                          result: str, details: Dict = None):
        """记录5步对齐过程"""
        message = f"第{segment_id}句对齐第{step}步({step_name}): {result}"
        
        context = {"segment_id": segment_id, "step": step, "step_name": step_name}
        if details:
            context.update(details)
            
        self.log("ALIGN", message, context=context)
    
    def log_processing_progress(self, current_step: str, progress: int, total: int = 100):
        """记录处理进度"""
        message = f"{current_step} - 进度: {progress}/{total}"
        context = {"current_step": current_step, "progress": progress, "total": total}
        self.log("PROCESS", message, context=context)
    
    def log_error_with_retry(self, operation: str, attempt: int, max_attempts: int, 
                            error: str, trace_id: str = None):
        """记录重试错误"""
        if attempt < max_attempts:
            message = f"{operation}失败,第{attempt}次重试中... 错误: {error}"
            level = "WARNING"
        else:
            message = f"{operation}重试{max_attempts}次后最终失败: {error}"
            level = "ERROR"
            
        self.log(level, message, trace_id)
    
    def get_logs(self, level_filter: str = None, limit: int = None) -> List[Dict]:
        """获取日志列表"""
        logs = list(self.logs)
        
        # 按级别过滤
        if level_filter:
            logs = [log for log in logs if log["level"] == level_filter]
        
        # 限制数量
        if limit:
            logs = logs[-limit:]
            
        return logs
    
    def get_formatted_logs(self, level_filter: str = None, limit: int = 100) -> List[str]:
        """获取格式化的日志字符串"""
        logs = self.get_logs(level_filter, limit)
        formatted = []
        
        for log in logs:
            timestamp = log["timestamp"]
            level = log["level"]
            message = log["message"]
            
            formatted_log = f"{timestamp} [{level}] {message}"
            
            if log.get("trace_id"):
                formatted_log += f" Trace-ID: {log['trace_id']}"
                
            formatted.append(formatted_log)
            
        return formatted
    
    def clear_logs(self):
        """清空日志"""
        self.logs.clear()
    
    def get_log_stats(self) -> Dict:
        """获取日志统计"""
        if not self.logs:
            return {"total": 0, "by_level": {}}
        
        by_level = {}
        for log in self.logs:
            level = log["level"]
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            "total": len(self.logs),
            "by_level": by_level,
            "latest": self.logs[-1] if self.logs else None
        }
    
    def export_logs(self, filename: str = None) -> str:
        """导出日志到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_translator_logs_{timestamp}.txt"
        
        formatted_logs = self.get_formatted_logs()
        log_content = "\n".join(formatted_logs)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(log_content)
            
        return filename