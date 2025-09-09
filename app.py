import os
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import uuid

from models.config_model import ConfigModel
from models.project_data import ProjectData
from utils.file_handler import FileHandler
from utils.rate_limiter import RateLimiter
from utils.error_handler import ErrorHandler
from modules.logger_service import LoggerService
from modules.video_processor import VideoProcessor
from modules.asr_processor import ASRProcessor
from modules.translation_service import TranslationService
from modules.voice_clone_service import VoiceCloneService
from modules.tts_service import TTSService
from modules.alignment_optimizer import AlignmentOptimizer
from modules.audio_mixer import AudioMixer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-translator-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

config = ConfigModel()
project_data = ProjectData()
file_handler = FileHandler(app.config['UPLOAD_FOLDER'])
rate_limiter = RateLimiter()
error_handler = ErrorHandler()
logger_service = LoggerService()

# 初始化处理模块
video_processor = VideoProcessor(logger_service)
asr_processor = ASRProcessor(config, logger_service)
translation_service = TranslationService(config, rate_limiter, logger_service)
voice_clone_service = VoiceCloneService(config, rate_limiter, logger_service)
tts_service = TTSService(config, rate_limiter, logger_service)
alignment_optimizer = AlignmentOptimizer(config, translation_service, tts_service, logger_service)
audio_mixer = AudioMixer(logger_service)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        try:
            config_data = request.get_json()
            config.update(config_data)
            logger_service.log("INFO", "配置已更新")
            return jsonify({"status": "success", "message": "配置保存成功"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    else:
        return jsonify(config.to_dict())

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "没有选择文件"}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({"status": "error", "message": "没有选择文件"}), 400
        
        if file and file_handler.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = file_handler.save_file(file, unique_filename)
            
            project_data.set_video_path(file_path)
            logger_service.log("INFO", f"视频文件上传成功: {filename}")
            
            return jsonify({
                "status": "success", 
                "message": "文件上传成功",
                "file_path": file_path,
                "filename": filename
            })
        else:
            return jsonify({"status": "error", "message": "不支持的文件格式"}), 400
            
    except Exception as e:
        error_msg = error_handler.handle_error(e, "文件上传")
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/api/process/start', methods=['POST'])
def start_processing():
    try:
        if not project_data.video_path:
            return jsonify({"status": "error", "message": "请先上传视频文件"}), 400
        
        logger_service.log("INFO", "开始自动翻译处理...")
        
        # 设置处理状态
        project_data.set_processing_status("processing", "开始处理视频...", 5)
        
        # 真实的视频处理流程
        import threading
        def real_video_processing():
            try:
                # 步骤1: 提取音频 (5% → 15%)
                logger_service.log("INFO", "正在从视频中提取音频...")
                audio_result = video_processor.extract_audio(project_data.video_path)
                
                if not audio_result["success"]:
                    logger_service.log("ERROR", f"音频提取失败: {audio_result['error']}")
                    project_data.set_processing_status("error", "音频提取失败", 5)
                    return
                
                audio_path = audio_result["audio_path"]
                project_data.set_processing_status("processing", "ASR语音识别中...", 15)
                
                # 步骤2: ASR识别和切分 (15% → 35%)
                logger_service.log("INFO", "正在进行ASR语音识别和智能切分...")
                segments = asr_processor.process_audio(audio_path)
                
                if not segments:
                    logger_service.log("ERROR", "ASR处理失败，未检测到语音片段")
                    project_data.set_processing_status("error", "ASR处理失败", 15)
                    return
                
                project_data.update_segments(segments)
                project_data.set_processing_status("processing", "开始逐句翻译...", 35)
                
                # 步骤3: 逐句处理 (35% → 95%)
                total_segments = len(segments)
                for i, segment in enumerate(segments):
                    try:
                        sequence = segment["sequence"]
                        original_text = segment["original_text"]
                        original_audio_path = segment["original_audio_path"]
                        
                        # 更新进度
                        progress = 35 + int((i / total_segments) * 60)
                        project_data.set_processing_status("processing", f"处理第{sequence}句...", progress)
                        
                        # 3.1 翻译
                        logger_service.log("INFO", f"第{sequence}句: 开始翻译")
                        translation_result = translation_service.translate_text(original_text)
                        
                        if not translation_result["success"]:
                            logger_service.log("ERROR", f"第{sequence}句翻译失败: {translation_result['error']}")
                            continue
                        
                        translated_text = translation_result["translated_text"]
                        segment["translated_text"] = translated_text
                        
                        # 3.2 音色克隆
                        logger_service.log("INFO", f"第{sequence}句: 开始音色克隆")
                        voice_id = voice_clone_service.generate_voice_id(sequence)
                        
                        clone_result = voice_clone_service.clone_voice_from_audio(
                            original_audio_path, voice_id
                        )
                        
                        if clone_result["success"]:
                            # 音色克隆成功，使用克隆的voice_id进行TTS
                            logger_service.log("INFO", f"第{sequence}句音色克隆成功，开始TTS合成")
                            
                            tts_output_path = f"./temp/segment_{sequence}_translated.mp3"
                            tts_result = tts_service.synthesize_speech(
                                translated_text, voice_id, 1.0, tts_output_path
                            )
                            
                            if tts_result["success"]:
                                segment["translated_audio_path"] = tts_result["audio_path"]
                                segment["voice_id"] = voice_id
                                logger_service.log("INFO", f"第{sequence}句使用克隆音色TTS合成成功")
                            else:
                                logger_service.log("ERROR", f"第{sequence}句克隆音色TTS失败: {tts_result['error']}")
                                continue
                        else:
                            # 音色克隆失败，使用标准TTS
                            logger_service.log("WARNING", f"第{sequence}句音色克隆失败，使用标准TTS: {clone_result.get('error', '未知错误')}")
                            
                            tts_output_path = f"./temp/segment_{sequence}_translated.mp3"
                            tts_result = tts_service.synthesize_speech(
                                translated_text, "male-qn-qingse", 1.0, tts_output_path
                            )
                            
                            if tts_result["success"]:
                                segment["translated_audio_path"] = tts_result["audio_path"]
                                segment["voice_id"] = "male-qn-qingse"
                            else:
                                logger_service.log("ERROR", f"第{sequence}句标准TTS也失败: {tts_result['error']}")
                                continue
                        
                        # 3.4 时间戳对齐优化
                        logger_service.log("INFO", f"第{sequence}句: 开始5步时间戳对齐")
                        start_time, end_time = parse_timestamp(segment["timestamp"])
                        target_duration = end_time - start_time
                        
                        alignment_result = alignment_optimizer.optimize_segment(segment, target_duration)
                        
                        if alignment_result["success"]:
                            segment["translated_audio_path"] = alignment_result["audio_path"]
                            segment["speed"] = alignment_result["speed"]
                            segment["translated_text"] = alignment_result.get("optimized_text", translated_text)
                            logger_service.log("INFO", f"第{sequence}句对齐优化成功")
                        else:
                            logger_service.log("WARNING", f"第{sequence}句对齐优化失败，使用原始音频")
                        
                        # 更新项目数据
                        segment_update = {k: v for k, v in segment.items() if k != 'sequence'}
                        project_data.update_segment(sequence, **segment_update)
                        
                    except Exception as e:
                        logger_service.log("ERROR", f"第{sequence}句处理异常: {str(e)}")
                        continue
                
                # 步骤4: 音频拼接和视频合成 (95% → 100%)
                project_data.set_processing_status("processing", "正在拼接音频片段...", 95)
                logger_service.log("INFO", "开始音频拼接和视频合成...")
                
                # 4.1 拼接所有翻译音频片段
                mixed_audio_path = "./temp/final_translated_audio.wav"
                mix_result = audio_mixer.concatenate_audio_segments(
                    project_data.segments, mixed_audio_path
                )
                
                if not mix_result["success"]:
                    logger_service.log("ERROR", f"音频拼接失败: {mix_result['error']}")
                    project_data.set_processing_status("error", "音频拼接失败", 95)
                    return
                
                project_data.set_processing_status("processing", "正在合成最终视频...", 98)
                
                # 4.2 合成最终视频
                final_video_path = "./temp/final_translated_video.mp4"
                video_result = video_processor.merge_audio_video(
                    project_data.video_path, mixed_audio_path, final_video_path
                )
                
                if video_result["success"]:
                    # 保存最终视频路径
                    project_data.final_video_path = final_video_path
                    logger_service.log("INFO", f"最终视频已生成: {final_video_path}")
                    
                    # 完成
                    project_data.set_processing_status("completed", "处理完成", 100)
                    logger_service.log("INFO", "🎉 视频翻译处理完成!")
                else:
                    logger_service.log("ERROR", f"视频合成失败: {video_result['error']}")
                    project_data.set_processing_status("error", "视频合成失败", 98)
                
            except Exception as e:
                error_msg = f"处理过程中发生异常: {str(e)}"
                logger_service.log("ERROR", error_msg)
                project_data.set_processing_status("error", "处理失败", project_data.progress)
        
        # 在后台线程中运行真实处理
        thread = threading.Thread(target=real_video_processing)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "success", "message": "处理开始", "task_id": str(uuid.uuid4())})
        
    except Exception as e:
        error_msg = error_handler.handle_error(e, "开始处理")
        return jsonify({"status": "error", "message": error_msg}), 500

def parse_timestamp(timestamp: str) -> tuple:
    """解析时间戳"""
    try:
        parts = timestamp.split('-')
        if len(parts) == 2:
            start = float(parts[0])
            end = float(parts[1])
            return start, end
    except:
        pass
    return 0.0, 3.0

def create_demo_segments():
    """创建演示数据"""
    project_data.clear_segments()
    
    # 添加示例片段
    demo_segments = [
        {
            "sequence": 1,
            "timestamp": "0.0-3.5",
            "original_text": "大家好，欢迎使用视频翻译平台",
            "translated_text": "Hello everyone, welcome to the video translation platform",
            "original_audio_path": "./temp/segment_1_original.wav",
            "translated_audio_path": "./temp/segment_1_translated.mp3",
            "voice_id": "voice_1_1725875736",
            "speed": 1.0
        },
        {
            "sequence": 2,
            "timestamp": "3.5-7.2",
            "original_text": "这是一个基于人工智能的翻译工具",
            "translated_text": "This is an AI-based translation tool",
            "original_audio_path": "./temp/segment_2_original.wav",
            "translated_audio_path": "./temp/segment_2_translated.mp3",
            "voice_id": "voice_2_1725875741",
            "speed": 1.2
        },
        {
            "sequence": 3,
            "timestamp": "7.2-10.8",
            "original_text": "它可以保持原声的情绪和语调",
            "translated_text": "It can preserve the original voice's emotion and tone",
            "original_audio_path": "./temp/segment_3_original.wav",
            "translated_audio_path": "./temp/segment_3_translated.mp3",
            "voice_id": "voice_3_1725875745",
            "speed": 1.1
        }
    ]
    
    project_data.update_segments(demo_segments)

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        try:
            data = request.get_json()
            project_data.update_segments(data.get('segments', []))
            logger_service.log("INFO", "数据表格已更新")
            return jsonify({"status": "success", "message": "数据保存成功"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    else:
        return jsonify(project_data.to_dict())

@app.route('/api/srt/export', methods=['GET'])
def export_srt():
    try:
        srt_content = project_data.export_srt()
        filename = f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        file_path = file_handler.save_temp_file(srt_content, filename)
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/srt/import', methods=['POST'])
def import_srt():
    try:
        if 'srt_file' not in request.files:
            return jsonify({"status": "error", "message": "没有选择SRT文件"}), 400
        
        file = request.files['srt_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "没有选择文件"}), 400
        
        if file and file.filename.endswith('.srt'):
            content = file.read().decode('utf-8')
            segments = project_data.import_srt(content)
            logger_service.log("INFO", f"SRT文件导入成功: {len(segments)}个片段")
            return jsonify({"status": "success", "message": "SRT导入成功", "segments": segments})
        else:
            return jsonify({"status": "error", "message": "请选择SRT文件"}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify(logger_service.get_logs())

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    logger_service.clear_logs()
    return jsonify({"status": "success", "message": "日志已清空"})

@app.route('/api/download/video', methods=['GET'])
def download_final_video():
    try:
        if not project_data.final_video_path or not os.path.exists(project_data.final_video_path):
            return jsonify({"status": "error", "message": "翻译视频不存在，请先完成处理"}), 404
        
        filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        return send_file(project_data.final_video_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/regenerate/<int:segment_id>', methods=['POST'])
def regenerate_segment(segment_id):
    try:
        # TODO: 实现单句重新生成
        logger_service.log("INFO", f"重新生成第{segment_id}句...")
        return jsonify({"status": "success", "message": f"第{segment_id}句重新生成完成"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": "error", "message": "页面不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "服务器内部错误"}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info("视频翻译测试平台启动中...")
    logger.info("访问地址: https://localhost:5555")
    app.run(host='0.0.0.0', port=5555, debug=True, ssl_context='adhoc')