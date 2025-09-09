import traceback
import logging
from datetime import datetime

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.last_errors = []
        
    def handle_error(self, error, context="未知操作", max_history=50):
        """处理错误并返回用户友好的错误消息"""
        self.error_count += 1
        timestamp = datetime.now().isoformat()
        
        # 获取错误堆栈
        error_trace = traceback.format_exc()
        
        # 记录详细错误信息
        error_info = {
            "timestamp": timestamp,
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "trace": error_trace
        }
        
        self.last_errors.append(error_info)
        
        # 保持错误历史记录在合理范围内
        if len(self.last_errors) > max_history:
            self.last_errors = self.last_errors[-max_history:]
        
        # 记录到日志
        self.logger.error(f"错误发生在 {context}: {str(error)}")
        self.logger.debug(f"错误堆栈: {error_trace}")
        
        # 返回用户友好的错误消息
        return self._get_user_friendly_message(error, context)
    
    def _get_user_friendly_message(self, error, context):
        """将技术错误转换为用户友好的消息"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # API相关错误
        if "connectionerror" in error_type.lower() or "timeout" in error_msg:
            return f"{context}失败: 网络连接超时，请检查网络连接"
        
        if "401" in error_msg or "unauthorized" in error_msg:
            return f"{context}失败: API密钥无效，请检查配置"
        
        if "403" in error_msg or "forbidden" in error_msg:
            return f"{context}失败: 访问被拒绝，请检查权限"
        
        if "429" in error_msg or "rate limit" in error_msg:
            return f"{context}失败: 请求过于频繁，请稍后重试"
        
        if "500" in error_msg or "internal server error" in error_msg:
            return f"{context}失败: 服务器内部错误，请稍后重试"
        
        # 文件相关错误
        if "filenotfounderror" in error_type.lower():
            return f"{context}失败: 找不到文件，请检查文件路径"
        
        if "permissionerror" in error_type.lower():
            return f"{context}失败: 权限不足，请检查文件权限"
        
        if "no space left" in error_msg:
            return f"{context}失败: 磁盘空间不足"
        
        # 数据格式错误
        if "json" in error_msg and ("decode" in error_msg or "parse" in error_msg):
            return f"{context}失败: 数据格式错误，请检查输入"
        
        if "valueerror" in error_type.lower():
            return f"{context}失败: 输入参数无效，请检查配置"
        
        # 通用错误
        if len(str(error)) > 100:
            return f"{context}失败: 处理过程中出现错误，请查看日志"
        
        return f"{context}失败: {str(error)}"
    
    def get_error_stats(self):
        """获取错误统计信息"""
        if not self.last_errors:
            return {
                "total_errors": self.error_count,
                "recent_errors": 0,
                "error_types": {},
                "contexts": {}
            }
        
        # 统计错误类型和上下文
        error_types = {}
        contexts = {}
        
        for error in self.last_errors:
            error_type = error["error_type"]
            context = error["context"]
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            contexts[context] = contexts.get(context, 0) + 1
        
        return {
            "total_errors": self.error_count,
            "recent_errors": len(self.last_errors),
            "error_types": error_types,
            "contexts": contexts,
            "last_error": self.last_errors[-1] if self.last_errors else None
        }
    
    def clear_history(self):
        """清空错误历史"""
        self.last_errors = []
        self.error_count = 0
    
    def get_recent_errors(self, limit=10):
        """获取最近的错误"""
        return self.last_errors[-limit:] if self.last_errors else []