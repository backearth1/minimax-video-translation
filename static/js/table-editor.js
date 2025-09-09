// 表格编辑器
class TableEditor {
    constructor() {
        this.currentEditingCell = null;
        this.bindEvents();
    }
    
    bindEvents() {
        // 延迟绑定，确保DOM已加载
        setTimeout(() => {
            this.bindEditableEvents();
        }, 100);
    }
    
    bindEditableEvents() {
        // 移除之前的事件监听器
        document.querySelectorAll('.editable-cell').forEach(cell => {
            cell.removeEventListener('dblclick', this.handleCellDoubleClick);
        });
        
        // 重新绑定双击编辑事件
        document.querySelectorAll('.editable-cell').forEach(cell => {
            cell.addEventListener('dblclick', (e) => this.handleCellDoubleClick(e));
        });
    }
    
    handleCellDoubleClick(e) {
        const cell = e.target;
        
        // 如果当前有正在编辑的单元格，先保存
        if (this.currentEditingCell && this.currentEditingCell !== cell) {
            this.saveCell(this.currentEditingCell);
        }
        
        this.editCell(cell);
    }
    
    editCell(cell) {
        if (cell.classList.contains('editing')) {
            return;
        }
        
        const originalText = cell.textContent.trim();
        const field = cell.dataset.field;
        const id = cell.dataset.id;
        
        cell.classList.add('editing');
        this.currentEditingCell = cell;
        
        // 根据字段类型创建不同的输入控件
        let inputElement;
        if (field === 'original_text' || field === 'translated_text') {
            inputElement = document.createElement('textarea');
            inputElement.rows = 3;
        } else {
            inputElement = document.createElement('input');
            inputElement.type = 'text';
        }
        
        inputElement.value = originalText;
        inputElement.className = 'form-control';
        
        // 清空单元格并添加输入框
        cell.innerHTML = '';
        cell.appendChild(inputElement);
        
        // 聚焦并选中文本
        inputElement.focus();
        inputElement.select();
        
        // 绑定事件
        inputElement.addEventListener('blur', () => {
            this.saveCell(cell);
        });
        
        inputElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.saveCell(cell);
            } else if (e.key === 'Escape') {
                this.cancelEdit(cell, originalText);
            }
        });
    }
    
    async saveCell(cell) {
        if (!cell.classList.contains('editing')) {
            return;
        }
        
        const inputElement = cell.querySelector('input, textarea');
        if (!inputElement) {
            return;
        }
        
        const newValue = inputElement.value.trim();
        const field = cell.dataset.field;
        const id = parseInt(cell.dataset.id);
        
        // 恢复单元格显示
        cell.classList.remove('editing');
        cell.textContent = newValue;
        this.currentEditingCell = null;
        
        // 保存到服务器
        try {
            await this.updateSegmentData(id, field, newValue);
            this.showCellSaveSuccess(cell);
        } catch (error) {
            this.showCellSaveError(cell, error.message);
            console.error('保存失败:', error);
        }
    }
    
    cancelEdit(cell, originalText) {
        cell.classList.remove('editing');
        cell.textContent = originalText;
        this.currentEditingCell = null;
    }
    
    async updateSegmentData(segmentId, field, value) {
        // 获取当前所有片段数据
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error('获取数据失败');
        }
        
        const data = await response.json();
        const segments = data.segments || [];
        
        // 更新指定片段的字段
        const segment = segments.find(s => s.sequence === segmentId);
        if (!segment) {
            throw new Error('未找到指定片段');
        }
        
        // 类型转换
        if (field === 'speed') {
            value = parseFloat(value) || 1.0;
            if (value < 0.5 || value > 2.0) {
                throw new Error('速度参数必须在0.5-2.0之间');
            }
        } else if (field === 'sequence') {
            value = parseInt(value) || 1;
        }
        
        segment[field] = value;
        
        // 提交更新
        const updateResponse = await fetch('/api/data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ segments })
        });
        
        if (!updateResponse.ok) {
            const errorData = await updateResponse.json();
            throw new Error(errorData.message || '保存失败');
        }
        
        // 添加日志
        if (window.app) {
            window.app.addLog('INFO', `第${segmentId}句${this.getFieldDisplayName(field)}已更新: ${value}`);
        }
    }
    
    getFieldDisplayName(field) {
        const fieldNames = {
            'timestamp': '时间戳',
            'original_text': '原文本',
            'translated_text': '翻译文本',
            'speed': '速度参数',
            'voice_id': '音色ID'
        };
        return fieldNames[field] || field;
    }
    
    showCellSaveSuccess(cell) {
        const originalBg = cell.style.backgroundColor;
        cell.style.backgroundColor = '#d4edda';
        cell.style.transition = 'background-color 0.3s ease';
        
        setTimeout(() => {
            cell.style.backgroundColor = originalBg;
        }, 1000);
    }
    
    showCellSaveError(cell, errorMessage) {
        const originalBg = cell.style.backgroundColor;
        cell.style.backgroundColor = '#f8d7da';
        cell.style.transition = 'background-color 0.3s ease';
        
        // 显示错误提示
        const tooltip = document.createElement('div');
        tooltip.className = 'alert alert-danger alert-dismissible fade show position-absolute';
        tooltip.style.cssText = 'top: 0; left: 0; z-index: 9999; font-size: 12px; padding: 0.25rem 0.5rem;';
        tooltip.innerHTML = `
            ${errorMessage}
            <button type="button" class="btn-close btn-close-sm" onclick="this.parentElement.remove()"></button>
        `;
        
        cell.style.position = 'relative';
        cell.appendChild(tooltip);
        
        setTimeout(() => {
            cell.style.backgroundColor = originalBg;
            if (tooltip.parentElement) {
                tooltip.remove();
            }
        }, 3000);
    }
    
    // 添加新行
    addNewRow() {
        const tbody = document.getElementById('segmentTableBody');
        const rows = tbody.querySelectorAll('tr');
        
        // 如果当前是空表格，清除占位符
        if (rows.length === 1 && rows[0].querySelector('td[colspan]')) {
            tbody.innerHTML = '';
        }
        
        const newSequence = rows.length + 1;
        const row = document.createElement('tr');
        row.className = 'segment-row';
        row.innerHTML = `
            <td>${newSequence}</td>
            <td class="editable-cell" data-field="timestamp" data-id="${newSequence}">0.0-3.0</td>
            <td class="speaker-cell"><span class="badge bg-secondary">未知</span></td>
            <td class="editable-cell" data-field="original_text" data-id="${newSequence}"></td>
            <td class="editable-cell" data-field="translated_text" data-id="${newSequence}"></td>
            <td class="audio-cell"><span class="text-muted">-</span></td>
            <td class="audio-cell"><span class="text-muted">-</span></td>
            <td class="editable-cell" data-field="speed" data-id="${newSequence}">1.0</td>
            <td class="editable-cell" data-field="voice_id" data-id="${newSequence}"></td>
            <td class="action-buttons">
                <button class="btn btn-sm btn-outline-primary" onclick="app.regenerateSegment(${newSequence})">
                    <i class="fas fa-redo"></i>生成
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="tableEditor.deleteRow(${newSequence})">
                    <i class="fas fa-trash"></i>删除
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
        this.bindEditableEvents();
        
        // 更新统计
        document.getElementById('segmentCount').textContent = tbody.querySelectorAll('tr').length;
        
        if (window.app) {
            window.app.addLog('INFO', `添加了新的第${newSequence}句`);
        }
    }
    
    // 删除行
    deleteRow(sequence) {
        if (confirm(`确定要删除第${sequence}句吗？`)) {
            const row = document.querySelector(`tr .editable-cell[data-id="${sequence}"]`)?.parentElement;
            if (row) {
                row.remove();
                
                // 重新编号
                this.renumberRows();
                
                if (window.app) {
                    window.app.addLog('INFO', `删除了第${sequence}句`);
                }
            }
        }
    }
    
    // 重新编号
    renumberRows() {
        const tbody = document.getElementById('segmentTableBody');
        const rows = tbody.querySelectorAll('tr');
        
        rows.forEach((row, index) => {
            const sequenceCell = row.querySelector('td:first-child');
            const editableCells = row.querySelectorAll('.editable-cell');
            
            const newSequence = index + 1;
            sequenceCell.textContent = newSequence;
            
            editableCells.forEach(cell => {
                cell.dataset.id = newSequence;
            });
            
            // 更新操作按钮
            const buttons = row.querySelectorAll('button');
            buttons.forEach(button => {
                if (button.onclick) {
                    const onclick = button.onclick.toString();
                    if (onclick.includes('regenerateSegment')) {
                        button.onclick = () => window.app.regenerateSegment(newSequence);
                    } else if (onclick.includes('deleteRow')) {
                        button.onclick = () => this.deleteRow(newSequence);
                    }
                }
            });
        });
        
        // 更新统计
        document.getElementById('segmentCount').textContent = rows.length;
    }
    
    // 批量编辑
    enableBatchEdit() {
        const checkboxes = this.addSelectionCheckboxes();
        
        // 显示批量操作工具栏
        this.showBatchToolbar();
    }
    
    addSelectionCheckboxes() {
        const rows = document.querySelectorAll('#segmentTableBody tr');
        const checkboxes = [];
        
        rows.forEach(row => {
            const firstCell = row.querySelector('td:first-child');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input me-2';
            firstCell.prepend(checkbox);
            checkboxes.push(checkbox);
        });
        
        return checkboxes;
    }
    
    showBatchToolbar() {
        const toolbar = document.createElement('div');
        toolbar.id = 'batchToolbar';
        toolbar.className = 'alert alert-info d-flex justify-content-between align-items-center';
        toolbar.innerHTML = `
            <div>
                <strong>批量编辑模式</strong> - 选择要编辑的行
            </div>
            <div>
                <button class="btn btn-sm btn-primary me-2" onclick="tableEditor.applyBatchEdit()">应用编辑</button>
                <button class="btn btn-sm btn-secondary" onclick="tableEditor.cancelBatchEdit()">取消</button>
            </div>
        `;
        
        const tableCard = document.querySelector('#segmentTable').closest('.card');
        tableCard.insertBefore(toolbar, tableCard.querySelector('.card-body'));
    }
    
    applyBatchEdit() {
        // TODO: 实现批量编辑逻辑
        console.log('批量编辑功能待实现');
        this.cancelBatchEdit();
    }
    
    cancelBatchEdit() {
        // 移除复选框
        document.querySelectorAll('#segmentTableBody input[type="checkbox"]').forEach(cb => cb.remove());
        
        // 移除工具栏
        const toolbar = document.getElementById('batchToolbar');
        if (toolbar) toolbar.remove();
    }
}

// 初始化表格编辑器
let tableEditor;
document.addEventListener('DOMContentLoaded', () => {
    tableEditor = new TableEditor();
    
    // 导出到全局作用域供HTML使用
    window.tableEditor = tableEditor;
});