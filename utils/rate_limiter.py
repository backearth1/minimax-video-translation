import time
from collections import defaultdict, deque
from threading import Lock

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = Lock()
        
        # API限流配置
        self.limits = {
            'llm': {'requests': 1, 'period': 1},      # LLM请求1次/秒
            'clone': {'requests': 1, 'period': 2},    # 克隆请求1次/2秒
            'tts': {'requests': 1, 'period': 2},      # TTS请求1次/2秒
            'default': {'requests': 10, 'period': 1}  # 默认限制
        }
    
    def can_make_request(self, api_type='default'):
        """检查是否可以发起请求"""
        with self.lock:
            now = time.time()
            limit_config = self.limits.get(api_type, self.limits['default'])
            
            # 清理过期的请求记录
            cutoff_time = now - limit_config['period']
            while self.requests[api_type] and self.requests[api_type][0] < cutoff_time:
                self.requests[api_type].popleft()
            
            # 检查是否超过限制
            if len(self.requests[api_type]) < limit_config['requests']:
                self.requests[api_type].append(now)
                return True
            
            return False
    
    def wait_for_availability(self, api_type='default'):
        """等待直到可以发起请求"""
        limit_config = self.limits.get(api_type, self.limits['default'])
        
        while not self.can_make_request(api_type):
            sleep_time = limit_config['period'] / limit_config['requests']
            time.sleep(sleep_time)
    
    def get_wait_time(self, api_type='default'):
        """获取需要等待的时间"""
        with self.lock:
            now = time.time()
            limit_config = self.limits.get(api_type, self.limits['default'])
            
            if len(self.requests[api_type]) < limit_config['requests']:
                return 0
            
            # 计算最早的请求何时过期
            oldest_request = self.requests[api_type][0]
            wait_time = oldest_request + limit_config['period'] - now
            
            return max(0, wait_time)
    
    def reset(self, api_type=None):
        """重置限流器"""
        with self.lock:
            if api_type:
                self.requests[api_type].clear()
            else:
                self.requests.clear()
    
    def get_status(self):
        """获取当前限流状态"""
        with self.lock:
            now = time.time()
            status = {}
            
            for api_type, limit_config in self.limits.items():
                cutoff_time = now - limit_config['period']
                
                # 清理过期记录
                if api_type in self.requests:
                    while self.requests[api_type] and self.requests[api_type][0] < cutoff_time:
                        self.requests[api_type].popleft()
                
                current_requests = len(self.requests.get(api_type, []))
                max_requests = limit_config['requests']
                
                status[api_type] = {
                    'current_requests': current_requests,
                    'max_requests': max_requests,
                    'period': limit_config['period'],
                    'available': current_requests < max_requests,
                    'wait_time': self.get_wait_time(api_type)
                }
            
            return status