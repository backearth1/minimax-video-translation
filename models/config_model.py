class ConfigModel:
    def __init__(self):
        self.api_endpoint = "https://api.minimaxi.com"
        self.group_id = "1747179187841536150"
        self.api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmnZzno4oiLCJVc2VyTmFtZSI6IuadnOejiiIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNzQ3MTc5MTg3ODQ5OTI0NzU4IiwiUGhvbmUiOiIxMzAyNTQ5MDQyMyIsIkdyb3VwSUQiOiIxNzQ3MTc5MTg3ODQxNTM2MTUwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiZGV2aW5AbWluaW1heGkuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjQtMTItMjMgMTE6NTE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.szVUN2AH7lJ9fQ3EYfzcLcamSCFAOye3Y6yO3Wj_tlNhnhBIYxEEMvZsVgH9mgOe6uhRczOqibmEMbVMUD_1DqtykrbD5klaB4_nhRnDl8fbaAf7m8B1OTRTUIiqgXRVglITenx3K_ugZ6teqiqypByJoLleHbZCSPWvy1-NaDiynb7qAsGzN1V6N4BOTNza1hL5PYdlrXLe2yjQv3YW8nOjQDIGCO1ZqnVBF0UghVaO4V-GZu1Z_0JnkLa7x_2ZXKXAe-LWhk9npwGFzQfLL3aH4oUzlsoEDGnuz3RZdZsFCe95MUiG8dCWfsxhVqlQ5GoFM3LQBAXuLZyqDpmSgg"
        self.source_language = "中文"
        self.target_language = "英语"
        self.llm_model = "MiniMax-Text-01"
        self.asr_model = "whisper-base"
        self.tts_model = "speech-2.5-hd-preview"
        self.asr_split_mode = "平衡模式"
        self.min_segment_duration = 1.5
        self.max_segment_duration = 8.0
        self.silence_threshold = 0.3
        
        # 支持的语言列表
        self.supported_languages = [
            "中文", "粤语", "英语", "西班牙语", "法语", "俄语", "德语", "葡萄牙语",
            "阿拉伯语", "意大利语", "日语", "韩语", "印尼语", "越南语", "土耳其语",
            "荷兰语", "乌克兰语", "泰语", "波兰语", "罗马尼亚语", "希腊语", "捷克语",
            "芬兰语", "印地语", "保加利亚语", "丹麦语", "希伯来语", "马来语", "波斯语",
            "斯洛伐克语", "瑞典语", "克罗地亚语", "菲律宾语", "匈牙利语", "挪威语",
            "斯洛文尼亚语", "加泰罗尼亚语", "尼诺斯克语", "泰米尔语", "阿非利卡语"
        ]
        
        # TTS模型选项
        self.tts_models = [
            "speech-2.5-hd-preview",
            "speech-02-hd", 
            "speech-01-hd"
        ]
        
        # ASR切分模式
        self.asr_split_modes = [
            "保守模式",
            "平衡模式", 
            "激进模式"
        ]
    
    def update(self, config_data):
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        return {
            "api_endpoint": self.api_endpoint,
            "group_id": self.group_id,
            "api_key": self.api_key,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "llm_model": self.llm_model,
            "asr_model": self.asr_model,
            "tts_model": self.tts_model,
            "asr_split_mode": self.asr_split_mode,
            "min_segment_duration": self.min_segment_duration,
            "max_segment_duration": self.max_segment_duration,
            "silence_threshold": self.silence_threshold,
            "supported_languages": self.supported_languages,
            "tts_models": self.tts_models,
            "asr_split_modes": self.asr_split_modes
        }
    
    def get_api_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def validate(self):
        if not self.group_id or not self.api_key:
            raise ValueError("Group ID和API Key不能为空")
        
        if self.target_language not in self.supported_languages:
            raise ValueError("不支持的目标语言")
            
        if self.tts_model not in self.tts_models:
            raise ValueError("不支持的TTS模型")
            
        if self.asr_split_mode not in self.asr_split_modes:
            raise ValueError("不支持的ASR切分模式")
            
        if not (1.0 <= self.min_segment_duration <= self.max_segment_duration <= 10.0):
            raise ValueError("时长参数设置不合理")
            
        return True