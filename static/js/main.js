// 主应用逻辑
class VideoTranslatorApp {
    constructor() {
        this.config = {};
        this.projectData = {};
        this.isProcessing = false;
        this.logs = [];
        
        this.init();
    }
    
    init() {
        this.loadConfig();
        this.bindEvents();
        this.setupLogRefresh();
    }
    
    // 加载配置
    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                this.config = await response.json();
                this.updateConfigUI();
            }
        } catch (error) {
            this.addLog('ERROR', '配置加载失败: ' + error.message);
        }
    }
    
    // 更新配置界面
    updateConfigUI() {
        document.getElementById('apiEndpoint').value = this.config.api_endpoint || 'https://api.minimaxi.com';
        document.getElementById('groupId').value = this.config.group_id || '';
        document.getElementById('apiKey').value = this.config.api_key || '';
        document.getElementById('sourceLanguage').value = this.config.source_language || '中文';
        document.getElementById('targetLanguage').value = this.config.target_language || '英语';
        document.getElementById('asrModel').value = this.config.asr_model || 'whisper-base';
        document.getElementById('ttsModel').value = this.config.tts_model || 'speech-2.5-hd-preview';
        document.getElementById('asrSplitMode').value = this.config.asr_split_mode || '平衡模式';
        document.getElementById('minDuration').value = this.config.min_segment_duration || 1.5;
        document.getElementById('maxDuration').value = this.config.max_segment_duration || 8.0;
        document.getElementById('silenceThreshold').value = this.config.silence_threshold || 0.3;
        
        // 填充目标语言选项
        if (this.config.supported_languages) {
            const targetSelect = document.getElementById('targetLanguage');
            targetSelect.innerHTML = '';
            this.config.supported_languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang;
                option.textContent = lang;
                targetSelect.appendChild(option);
            });
            targetSelect.value = this.config.target_language;
        }
    }
    
    // 绑定事件
    bindEvents() {
        // 保存配置
        document.getElementById('saveConfig').addEventListener('click', () => {
            this.saveConfig();
        });
        
        // 自动保存配置 - 监听配置项变化
        this.bindAutoSaveEvents();
        
        // 视频上传
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.uploadVideo();
        });
        
        // 开始AI翻译处理
        document.getElementById('startProfessionalProcessing').addEventListener('click', () => {
            this.startProfessionalProcessing();
        });
        
        // 导入SRT
        document.getElementById('importSrt').addEventListener('click', () => {
            document.getElementById('srtFileInput').click();
        });
        
        document.getElementById('srtFileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.importSRT(e.target.files[0]);
            }
        });
        
        // 导出SRT
        document.getElementById('exportSrt').addEventListener('click', () => {
            this.exportSRT();
        });
        
        // 重置数据
        document.getElementById('resetData').addEventListener('click', () => {
            this.resetData();
        });
        
        // 清空日志
        document.getElementById('clearLogs').addEventListener('click', () => {
            this.clearLogs();
        });
        
        // 检查AI模型状态
        document.getElementById('checkModels').addEventListener('click', () => {
            this.checkModelsStatus();
        });
        
        // 人工合成
        document.getElementById('manualSynth').addEventListener('click', () => {
            this.manualSynthesize();
        });
        
        // 视频文件选择变化
        document.getElementById('videoFile').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.previewVideo(e.target.files[0]);
            }
        });
        
        // 下载翻译视频
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadVideo();
        });
    }
    
    // 绑定自动保存事件
    bindAutoSaveEvents() {
        const configFields = [
            'apiEndpoint', 'groupId', 'apiKey', 'sourceLanguage', 'targetLanguage',
            'asrModel', 'ttsModel', 'asrSplitMode', 'minDuration', 'maxDuration', 'silenceThreshold'
        ];
        
        configFields.forEach(fieldId => {
            const element = document.getElementById(fieldId);
            if (element) {
                // 对于下拉框使用change事件，对于输入框使用blur事件
                const eventType = element.tagName === 'SELECT' ? 'change' : 'blur';
                element.addEventListener(eventType, () => {
                    this.autoSaveConfig();
                });
            }
        });
    }
    
    // 自动保存配置（延迟保存，避免频繁调用）
    autoSaveConfig() {
        // 清除之前的定时器
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
        }
        
        // 显示保存中状态
        this.updateConfigStatus('保存中...', 'text-warning');
        
        // 延迟500ms保存，避免用户快速切换时频繁保存
        this.autoSaveTimer = setTimeout(() => {
            this.saveConfig(true); // 传入true表示自动保存，不显示成功提示
        }, 500);
    }
    
    // 保存配置
    async saveConfig(isAutoSave = false) {
        const configData = {
            api_endpoint: document.getElementById('apiEndpoint').value,
            group_id: document.getElementById('groupId').value,
            api_key: document.getElementById('apiKey').value,
            source_language: document.getElementById('sourceLanguage').value,
            target_language: document.getElementById('targetLanguage').value,
            asr_model: document.getElementById('asrModel').value,
            tts_model: document.getElementById('ttsModel').value,
            asr_split_mode: document.getElementById('asrSplitMode').value,
            min_segment_duration: parseFloat(document.getElementById('minDuration').value),
            max_segment_duration: parseFloat(document.getElementById('maxDuration').value),
            silence_threshold: parseFloat(document.getElementById('silenceThreshold').value)
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configData)
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                this.config = configData;
                if (!isAutoSave) {
                    this.addLog('INFO', '配置保存成功');
                    this.showNotification('配置保存成功', 'success');
                } else {
                    this.addLog('DEBUG', '配置已自动保存');
                }
                this.updateConfigStatus('已保存', 'text-success');
            } else {
                this.addLog('ERROR', '配置保存失败: ' + result.message);
                if (!isAutoSave) {
                    this.showNotification('配置保存失败: ' + result.message, 'error');
                }
                this.updateConfigStatus('保存失败', 'text-danger');
            }
        } catch (error) {
            this.addLog('ERROR', '配置保存异常: ' + error.message);
            if (!isAutoSave) {
                this.showNotification('配置保存异常', 'error');
            }
            this.updateConfigStatus('保存异常', 'text-danger');
        }
    }
    
    // 更新配置状态指示器
    updateConfigStatus(text, className = '') {
        const statusElement = document.getElementById('configStatus');
        if (statusElement) {
            statusElement.textContent = text;
            statusElement.className = `text-muted d-block mt-1 ${className}`;
            
            // 3秒后恢复为默认状态
            if (className !== '') {
                setTimeout(() => {
                    statusElement.textContent = '配置会自动保存';
                    statusElement.className = 'text-muted d-block mt-1';
                }, 3000);
            }
        }
    }
    
    // 预览视频
    previewVideo(file) {
        const video = document.getElementById('originalVideo');
        const placeholder = document.getElementById('videoPlaceholder');
        
        const url = URL.createObjectURL(file);
        video.src = url;
        video.style.display = 'block';
        placeholder.style.display = 'none';
        
        this.addLog('INFO', `视频文件已选择: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    }
    
    // 上传视频
    async uploadVideo() {
        const fileInput = document.getElementById('videoFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showNotification('请先选择视频文件', 'warning');
            return;
        }
        
        // 检查文件大小（5分钟视频约300MB）
        if (file.size > 300 * 1024 * 1024) {
            this.showNotification('文件过大，请选择5分钟以内的视频', 'warning');
            return;
        }
        
        const formData = new FormData();
        formData.append('video', file);
        
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.innerHTML = '<span class="loading"></span>上传中...';
        uploadBtn.disabled = true;
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                this.addLog('INFO', '视频上传成功: ' + result.filename);
                this.showNotification('视频上传成功', 'success');
                
                // 启用AI翻译按钮
                document.getElementById('startProfessionalProcessing').disabled = false;
            } else {
                this.addLog('ERROR', '视频上传失败: ' + result.message);
                this.showNotification('视频上传失败: ' + result.message, 'error');
            }
        } catch (error) {
            this.addLog('ERROR', '视频上传异常: ' + error.message);
            this.showNotification('视频上传异常', 'error');
        } finally {
            uploadBtn.innerHTML = '<i class=\"fas fa-upload\"></i> 上传';
            uploadBtn.disabled = false;
        }
    }
    
    // 开始专业AI处理
    async startProfessionalProcessing() {
        if (this.isProcessing) {
            this.showNotification('正在处理中，请稍候...', 'info');
            return;
        }
        
        // 验证配置
        if (!this.config.group_id || !this.config.api_key) {
            this.showNotification('请先配置Group ID和API Key', 'warning');
            return;
        }
        
        this.isProcessing = true;
        const startBtn = document.getElementById('startProfessionalProcessing');
        startBtn.innerHTML = '<span class=\"loading\"></span>专业AI处理中...';
        startBtn.disabled = true;
        
        // 禁用AI翻译按钮，避免重复处理
        
        try {
            const response = await fetch('/api/process/professional', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                this.addLog('INFO', '🚀 专业AI翻译处理已开始');
                this.showNotification('专业AI处理已开始，使用Demucs+Whisper+pyannote', 'info');
                this.startProgressMonitoring();
            } else {
                this.addLog('ERROR', '专业处理启动失败: ' + result.message);
                this.showNotification('专业处理启动失败: ' + result.message, 'error');
                this.isProcessing = false;
            }
        } catch (error) {
            this.addLog('ERROR', '专业处理启动异常: ' + error.message);
            this.showNotification('专业处理启动异常', 'error');
            this.isProcessing = false;
        } finally {
            if (!this.isProcessing) {
                startBtn.innerHTML = '<i class=\"fas fa-star\"></i> 开始AI翻译';
                startBtn.disabled = false;
            }
        }
    }
    
    // 进度监控
    startProgressMonitoring() {
        const progressInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/data');
                if (response.ok) {
                    const data = await response.json();
                    this.updateProgress(data);
                    
                    if (data.processing_status === 'completed' || data.processing_status === 'error') {
                        clearInterval(progressInterval);
                        this.isProcessing = false;
                        
                        const professionalBtn = document.getElementById('startProfessionalProcessing');
                        professionalBtn.innerHTML = '<i class=\"fas fa-star\"></i> 开始AI翻译';
                        professionalBtn.disabled = false;
                        
                        if (data.processing_status === 'completed') {
                            this.addLog('INFO', '自动翻译处理完成');
                            this.showNotification('翻译完成！', 'success');
                            
                            // 显示下载按钮
                            document.getElementById('downloadBtn').style.display = 'inline-block';
                            
                            // 显示翻译后的视频预览
                            this.showTranslatedVideoPreview();
                        }
                    }
                }
            } catch (error) {
                console.error('进度监控错误:', error);
            }
        }, 2000);
    }
    
    // 更新进度
    updateProgress(data) {
        const progressBar = document.getElementById('progressBar');
        const currentStep = document.getElementById('currentStep');
        
        progressBar.style.width = `${data.progress || 0}%`;
        progressBar.textContent = `${data.progress || 0}%`;
        
        currentStep.textContent = data.current_step || '等待中...';
        
        // 更新数据表格
        if (data.segments && data.segments.length > 0) {
            this.updateSegmentTable(data.segments);
        }
        
        // 更新背景音频预览
        this.updateBackgroundAudio(data);
        
        // 更新统计信息
        document.getElementById('segmentCount').textContent = data.segment_count || 0;
    }
    
    // 更新音频预览
    updateBackgroundAudio(data) {
        // 更新人声预览
        this.updateAudioPreview('vocals', data.vocals_audio_available, data.vocals_audio_path, '分离');
        
        // 更新背景音预览
        this.updateAudioPreview('background', data.background_audio_available, data.background_audio_path, '分离');
        
        // 更新合成翻译人声预览
        this.updateAudioPreview('synthesized', data.synthesized_audio_available, data.synthesized_audio_path, '合成');
        
        // 更新最终混合音频预览
        this.updateAudioPreview('finalMixed', data.final_mixed_available, data.final_mixed_path, '混合');
    }
    
    // 通用音频预览更新函数
    updateAudioPreview(type, available, audioPath, actionText) {
        const player = document.getElementById(`${type}AudioPlayer`) || document.getElementById(`${type}VocalsPlayer`) || document.getElementById(`${type}Player`);
        const status = document.getElementById(`${type}AudioStatus`) || document.getElementById(`${type}VocalsStatus`) || document.getElementById(`${type}Status`);
        
        if (!player || !status) return;
        
        if (available && audioPath) {
            const audioSrc = `/api/audio/${encodeURIComponent(audioPath)}`;
            player.src = audioSrc;
            status.innerHTML = `<span class="text-success"><i class="fas fa-check-circle"></i> 已${actionText}</span>`;
            this.addLog('DEBUG', `${type}音频已加载: ${audioPath}`);
            
            // 绘制波形
            this.drawWaveform(player, `${type}Waveform`);
        } else if (audioPath) {
            status.innerHTML = `<span class="text-warning"><i class="fas fa-exclamation-triangle"></i> ${actionText}中...</span>`;
        } else {
            status.innerHTML = `<span class="text-muted"><i class="fas fa-minus-circle"></i> 未${actionText}</span>`;
        }
    }
    
    // 绘制波形图
    drawWaveform(audioElement, canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !audioElement.src) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 绘制简化的波形图（模拟）
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        const centerY = height / 2;
        const segments = 100;
        
        for (let i = 0; i < segments; i++) {
            const x = (i / segments) * width;
            // 生成模拟波形数据
            const amplitude = Math.random() * 0.8 + 0.2;
            const y = centerY + (Math.sin(i * 0.1) * amplitude * (height / 4));
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // 绘制中心线
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
        
        // 添加标签
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        ctx.fillText('音频波形', 5, 15);
    }
    
    // 更新片段表格
    updateSegmentTable(segments) {
        const tbody = document.getElementById('segmentTableBody');
        
        if (segments.length === 0) {
            tbody.innerHTML = '<tr><td colspan=\"10\" class=\"text-center text-muted\">暂无数据</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        segments.forEach((segment, index) => {
            const row = document.createElement('tr');
            row.className = 'segment-row';
            row.innerHTML = `
                <td>${segment.sequence}</td>
                <td class=\"editable-cell\" data-field=\"timestamp\" data-id=\"${segment.sequence}\">${segment.timestamp}</td>
                <td class=\"speaker-cell\">
                    <span class=\"badge bg-secondary\">${segment.speaker_id || '未知'}</span>
                </td>
                <td class=\"editable-cell\" data-field=\"original_text\" data-id=\"${segment.sequence}\">${segment.original_text}</td>
                <td class=\"editable-cell\" data-field=\"translated_text\" data-id=\"${segment.sequence}\">${segment.translated_text}</td>
                <td class=\"audio-cell\">
                    ${segment.original_audio_path ? `<audio controls style="width: 100%;"><source src="/api/audio/${encodeURIComponent(segment.original_audio_path)}" type="audio/wav">不支持音频播放</audio>` : '<span class="text-muted">-</span>'}
                </td>
                <td class=\"audio-cell\">
                    ${segment.translated_audio_path ? `<audio controls style="width: 100%;"><source src="/api/audio/${encodeURIComponent(segment.translated_audio_path)}" type="audio/mpeg">不支持音频播放</audio>` : '<span class="text-muted">-</span>'}
                </td>
                <td class=\"editable-cell\" data-field=\"speed\" data-id=\"${segment.sequence}\">${segment.speed}</td>
                <td class=\"editable-cell\" data-field=\"voice_id\" data-id=\"${segment.sequence}\">${segment.voice_id}</td>
                <td class=\"action-buttons\">
                    <button class=\"btn btn-sm btn-outline-primary\" onclick=\"app.regenerateSegment(${segment.sequence})\">
                        <i class=\"fas fa-redo\"></i>生成
                    </button>
                </td>
            `;
            tbody.appendChild(row);
        });
        
        // 重新绑定编辑事件
        window.tableEditor.bindEditableEvents();
    }
    
    // 重新生成片段
    async regenerateSegment(segmentId) {
        try {
            this.addLog('INFO', `开始重新生成第${segmentId}句...`);
            
            const response = await fetch(`/api/regenerate/${segmentId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                this.addLog('INFO', result.message);
                this.showNotification(`第${segmentId}句重新生成完成`, 'success');
            } else {
                this.addLog('ERROR', '重新生成失败: ' + result.message);
                this.showNotification('重新生成失败', 'error');
            }
        } catch (error) {
            this.addLog('ERROR', '重新生成异常: ' + error.message);
            this.showNotification('重新生成异常', 'error');
        }
    }
    
    // 导入SRT
    async importSRT(file) {
        const formData = new FormData();
        formData.append('srt_file', file);
        
        try {
            const response = await fetch('/api/srt/import', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.status === 'success') {
                this.addLog('INFO', `SRT文件导入成功: ${result.segments.length}个片段`);
                this.showNotification('SRT导入成功', 'success');
                this.updateSegmentTable(result.segments);
            } else {
                this.addLog('ERROR', 'SRT导入失败: ' + result.message);
                this.showNotification('SRT导入失败', 'error');
            }
        } catch (error) {
            this.addLog('ERROR', 'SRT导入异常: ' + error.message);
            this.showNotification('SRT导入异常', 'error');
        }
    }
    
    // 导出SRT
    async exportSRT() {
        try {
            const response = await fetch('/api/srt/export');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `translation_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.srt`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.addLog('INFO', 'SRT文件导出成功');
                this.showNotification('SRT导出成功', 'success');
            } else {
                this.showNotification('SRT导出失败', 'error');
            }
        } catch (error) {
            this.addLog('ERROR', 'SRT导出异常: ' + error.message);
            this.showNotification('SRT导出异常', 'error');
        }
    }
    
    // 重置数据
    resetData() {
        if (confirm('确定要重置所有数据吗？此操作不可撤销。')) {
            document.getElementById('segmentTableBody').innerHTML = '<tr><td colspan=\"10\" class=\"text-center text-muted\">暂无数据</td></tr>';
            document.getElementById('segmentCount').textContent = '0';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('progressBar').textContent = '0%';
            document.getElementById('currentStep').textContent = '等待开始...';
            
            this.addLog('INFO', '数据已重置');
            this.showNotification('数据已重置', 'info');
        }
    }
    
    // 检查AI模型状态
    async checkModelsStatus() {
        try {
            this.addLog('INFO', '正在检查AI模型状态...');
            const checkBtn = document.getElementById('checkModels');
            checkBtn.innerHTML = '<span class="loading"></span>检查中...';
            checkBtn.disabled = true;
            
            const response = await fetch('/api/models/status');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.addLog('INFO', 'AI模型状态检查完成，详细信息请查看上方日志');
                this.showNotification('模型状态检查完成', 'success');
                
                // 显示简要状态
                const models = result.models;
                let statusSummary = '';
                for (const [modelType, info] of Object.entries(models)) {
                    const status = info.available ? '✅ 已就绪' : '❌ 需下载';
                    statusSummary += `${info.description}: ${status}\\n`;
                }
                
                if (statusSummary) {
                    this.addLog('INFO', `📋 模型状态概览:\\n${statusSummary}`);
                }
            } else {
                this.addLog('ERROR', '模型状态检查失败: ' + result.message);
                this.showNotification('模型状态检查失败', 'error');
            }
        } catch (error) {
            this.addLog('ERROR', 'AI模型状态检查异常: ' + error.message);
            this.showNotification('模型状态检查异常', 'error');
        } finally {
            const checkBtn = document.getElementById('checkModels');
            checkBtn.innerHTML = '<i class="fas fa-search"></i> 检查AI模型';
            checkBtn.disabled = false;
        }
    }
    
    // 人工合成
    manualSynthesize() {
        this.addLog('INFO', '开始人工音频合成...');
        this.showNotification('功能开发中...', 'info');
    }
    
    // 显示翻译后的视频预览
    async showTranslatedVideoPreview() {
        try {
            // 使用专用的预览接口
            const response = await fetch('/api/video/preview');
            if (response.ok) {
                const blob = await response.blob();
                const videoUrl = URL.createObjectURL(blob);
                
                const translatedVideo = document.getElementById('translatedVideo');
                const resultPlaceholder = document.getElementById('resultPlaceholder');
                
                translatedVideo.src = videoUrl;
                translatedVideo.style.display = 'block';
                resultPlaceholder.style.display = 'none';
                
                this.addLog('INFO', '翻译后视频预览已加载');
                
                // 清理URL对象（当视频元素被销毁时）
                translatedVideo.addEventListener('loadstart', () => {
                    if (translatedVideo.previousSrc) {
                        URL.revokeObjectURL(translatedVideo.previousSrc);
                    }
                    translatedVideo.previousSrc = videoUrl;
                });
            } else {
                const errorText = await response.text();
                this.addLog('WARNING', `无法加载翻译后视频预览: ${response.status}`);
                console.error('Preview error:', errorText);
            }
        } catch (error) {
            this.addLog('ERROR', '加载视频预览失败: ' + error.message);
            console.error('Preview exception:', error);
        }
    }
    
    // 下载视频
    async downloadVideo() {
        try {
            const response = await fetch('/api/download/video');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `translated_video_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.mp4`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.addLog('INFO', '翻译视频下载成功');
                this.showNotification('视频下载成功', 'success');
            } else {
                const errorData = await response.json();
                this.showNotification('下载失败: ' + errorData.message, 'error');
            }
        } catch (error) {
            this.addLog('ERROR', '视频下载异常: ' + error.message);
            this.showNotification('视频下载异常', 'error');
        }
    }
    
    // 设置日志刷新
    setupLogRefresh() {
        setInterval(async () => {
            try {
                const response = await fetch('/api/logs');
                if (response.ok) {
                    const logs = await response.json();
                    this.updateLogs(logs);
                }
            } catch (error) {
                console.error('日志刷新失败:', error);
            }
        }, 3000);
    }
    
    // 更新日志
    updateLogs(logs) {
        const container = document.getElementById('logContainer');
        if (logs.length === 0) {
            container.innerHTML = '<div class=\"text-muted\">暂无日志</div>';
            return;
        }
        
        // 检查用户是否已经手动滚动到其他位置
        const isAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 5;
        
        const logHtml = logs.map(log => {
            const className = `log-entry ${log.level.toLowerCase()}`;
            let message = `${log.timestamp} [${log.level}] ${log.message}`;
            if (log.trace_id) {
                message += ` Trace-ID: ${log.trace_id}`;
            }
            return `<div class=\"${className}\">${message}</div>`;
        }).join('');
        
        container.innerHTML = logHtml;
        
        // 只有当用户在底部时才自动滚动，否则保持当前位置
        if (isAtBottom) {
            container.scrollTop = container.scrollHeight;
        }
    }
    
    // 添加日志
    addLog(level, message) {
        const timestamp = new Date().toLocaleString();
        const log = { timestamp, level, message };
        this.logs.push(log);
        
        // 本地日志显示
        const container = document.getElementById('logContainer');
        
        // 检查用户是否已经手动滚动到其他位置
        const isAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 5;
        
        const className = `log-entry ${level.toLowerCase()}`;
        const logElement = document.createElement('div');
        logElement.className = className;
        logElement.textContent = `${timestamp} [${level}] ${message}`;
        
        container.appendChild(logElement);
        
        // 只有当用户在底部时才自动滚动，否则保持当前位置
        if (isAtBottom) {
            container.scrollTop = container.scrollHeight;
        }
        
        // 保持日志数量在合理范围
        if (this.logs.length > 500) {
            this.logs = this.logs.slice(-400);
        }
    }
    
    // 清空日志
    async clearLogs() {
        try {
            const response = await fetch('/api/logs/clear', {
                method: 'POST'
            });
            
            if (response.ok) {
                document.getElementById('logContainer').innerHTML = '<div class=\"text-muted\">日志已清空</div>';
                this.logs = [];
                this.showNotification('日志已清空', 'info');
            }
        } catch (error) {
            this.showNotification('清空日志失败', 'error');
        }
    }
    
    // 显示通知
    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'info'} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type=\"button\" class=\"btn-close\" data-bs-dismiss=\"alert\"></button>
        `;
        
        document.body.appendChild(notification);
        
        // 3秒后自动移除
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
}

// 初始化应用
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new VideoTranslatorApp();
});