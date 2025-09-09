import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

class FileHandler:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.allowed_extensions = {
            'video': {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'},
            'audio': {'mp3', 'wav', 'aac', 'flac', 'ogg'},
            'subtitle': {'srt', 'ass', 'ssa', 'vtt'}
        }
        
        # 确保目录存在
        os.makedirs(upload_folder, exist_ok=True)
        
    def allowed_file(self, filename, file_type='video'):
        if '.' not in filename:
            return False
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in self.allowed_extensions.get(file_type, set())
    
    def save_file(self, file, filename=None):
        if filename is None:
            filename = secure_filename(file.filename)
            
        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(self.upload_folder, unique_filename)
        
        file.save(file_path)
        return file_path
    
    def save_temp_file(self, content, filename):
        """保存临时文件"""
        file_path = os.path.join(self.upload_folder, filename)
        
        if isinstance(content, str):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(file_path, 'wb') as f:
                f.write(content)
                
        return file_path
    
    def delete_file(self, file_path):
        """删除文件"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False
    
    def cleanup_old_files(self, max_age_hours=24):
        """清理超过指定时间的文件"""
        now = datetime.now().timestamp()
        max_age = max_age_hours * 3600
        
        deleted_count = 0
        for filename in os.listdir(self.upload_folder):
            file_path = os.path.join(self.upload_folder, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getctime(file_path)
                if file_age > max_age:
                    if self.delete_file(file_path):
                        deleted_count += 1
                        
        return deleted_count
    
    def get_file_info(self, file_path):
        """获取文件信息"""
        if not os.path.exists(file_path):
            return None
            
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.split('.')[-1].lower() if '.' in file_path else ""
        }
    
    def create_project_folder(self, project_id):
        """为项目创建专用文件夹"""
        project_folder = os.path.join(self.upload_folder, f"project_{project_id}")
        os.makedirs(project_folder, exist_ok=True)
        
        # 创建子文件夹
        subfolders = ['original', 'processed', 'audio', 'segments', 'output']
        for subfolder in subfolders:
            os.makedirs(os.path.join(project_folder, subfolder), exist_ok=True)
            
        return project_folder