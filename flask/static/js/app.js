/* TAS MCR-ALS 分析平台 JavaScript 工具函数 */

// 全局变量
window.TASApp = {
    currentSessionId: null,
    analysisStatus: null,
    uploadedFile: null,
    wsConnection: null
};

// 工具函数
const Utils = {
    // 格式化文件大小
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // 格式化日期时间
    formatDateTime: function(dateString) {
        if (!dateString) return '未知';
        const date = new Date(dateString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },

    // 延迟函数
    delay: function(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    // 生成UUID
    generateUUID: function() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    },

    // 验证文件类型
    validateFile: function(file) {
        const allowedTypes = ['.csv', '.txt', '.dat'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            return { valid: false, message: '不支持的文件类型。请上传 CSV、TXT 或 DAT 文件。' };
        }
        
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            return { valid: false, message: '文件太大。请上传小于 50MB 的文件。' };
        }
        
        return { valid: true };
    },

    // 显示Toast消息
    showToast: function(message, type = 'info', duration = 3000) {
        // 移除现有的toast容器
        let existingContainer = document.getElementById('toastContainer');
        if (existingContainer) {
            existingContainer.remove();
        }

        // 创建toast容器
        const toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);

        // 设置toast颜色
        const bgColor = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type] || 'bg-info';

        // 创建toast HTML
        const toastHtml = `
            <div class="toast align-items-center text-white ${bgColor} border-0" 
                 role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                            data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        toastContainer.innerHTML = toastHtml;

        // 显示toast
        const toastElement = toastContainer.querySelector('.toast');
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: duration
        });
        toast.show();

        // 自动移除容器
        toastElement.addEventListener('hidden.bs.toast', function() {
            toastContainer.remove();
        });
    },

    // 显示加载遮罩
    showLoading: function(message = '处理中...') {
        // 移除现有的加载遮罩
        const existingOverlay = document.getElementById('loadingOverlay');
        if (existingOverlay) {
            existingOverlay.remove();
        }

        const overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div>${message}</div>
            </div>
        `;
        document.body.appendChild(overlay);
    },

    // 隐藏加载遮罩
    hideLoading: function() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    },

    // 复制到剪贴板
    copyToClipboard: function(text) {
        if (navigator.clipboard) {
            return navigator.clipboard.writeText(text).then(() => {
                this.showToast('已复制到剪贴板', 'success');
                return true;
            }).catch(err => {
                console.error('复制失败:', err);
                this.showToast('复制失败', 'error');
                return false;
            });
        } else {
            // 降级方案
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                const successful = document.execCommand('copy');
                document.body.removeChild(textArea);
                if (successful) {
                    this.showToast('已复制到剪贴板', 'success');
                    return true;
                } else {
                    this.showToast('复制失败', 'error');
                    return false;
                }
            } catch (err) {
                document.body.removeChild(textArea);
                console.error('复制失败:', err);
                this.showToast('复制失败', 'error');
                return false;
            }
        }
    }
};

// 文件上传处理
const FileUpload = {
    // 初始化拖拽上传
    initDragAndDrop: function(dropAreaId, fileInputId, callbacks = {}) {
        const dropArea = document.getElementById(dropAreaId);
        const fileInput = document.getElementById(fileInputId);
        
        if (!dropArea || !fileInput) return;

        // 防止默认拖拽行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // 高亮效果
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });

        // 处理文件拖拽
        dropArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFiles(files, callbacks);
            }
        }, false);

        // 点击上传
        dropArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFiles(e.target.files, callbacks);
            }
        });
    },

    // 防止默认行为
    preventDefaults: function(e) {
        e.preventDefault();
        e.stopPropagation();
    },

    // 处理文件
    handleFiles: function(files, callbacks = {}) {
        if (files.length === 0) return;
        
        const file = files[0]; // 只处理第一个文件
        
        // 验证文件
        const validation = Utils.validateFile(file);
        if (!validation.valid) {
            Utils.showToast(validation.message, 'error');
            if (callbacks.onError) callbacks.onError(validation.message);
            return;
        }

        // 更新UI
        if (callbacks.onSelect) callbacks.onSelect(file);
        
        // 存储文件信息
        window.TASApp.uploadedFile = file;
    },

    // 上传文件
    uploadFile: function(file, uploadUrl, onProgress, onSuccess, onError) {
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();

        // 进度监听
        if (onProgress) {
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    onProgress(percentComplete);
                }
            });
        }

        // 完成监听
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (onSuccess) onSuccess(response);
                } catch (e) {
                    if (onError) onError('响应解析失败');
                }
            } else {
                if (onError) onError(`上传失败: ${xhr.statusText}`);
            }
        });

        // 错误监听
        xhr.addEventListener('error', () => {
            if (onError) onError('网络错误');
        });

        xhr.open('POST', uploadUrl);
        xhr.send(formData);

        return xhr;
    }
};

// 分析管理
const AnalysisManager = {
    // 开始分析
    startAnalysis: function(sessionId, parameters, onProgress, onComplete, onError) {
        window.TASApp.currentSessionId = sessionId;
        window.TASApp.analysisStatus = 'running';

        // 发送分析请求
        fetch('/api/run_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                ...parameters
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 开始轮询进度
                this.pollProgress(sessionId, onProgress, onComplete, onError);
            } else {
                if (onError) onError(data.message || '启动分析失败');
            }
        })
        .catch(error => {
            console.error('启动分析失败:', error);
            if (onError) onError('网络错误');
        });
    },

    // 轮询进度
    pollProgress: function(sessionId, onProgress, onComplete, onError) {
        const poll = () => {
            if (window.TASApp.analysisStatus !== 'running') {
                return; // 停止轮询
            }

            fetch(`/api/get_progress/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const sessionInfo = data.session_info;
                    
                    if (sessionInfo.status === 'completed') {
                        window.TASApp.analysisStatus = 'completed';
                        if (onComplete) onComplete(sessionInfo);
                    } else if (sessionInfo.status === 'error') {
                        window.TASApp.analysisStatus = 'error';
                        if (onError) onError(sessionInfo.error_message || '分析过程中发生错误');
                    } else {
                        // 继续运行，更新进度
                        if (onProgress) onProgress(sessionInfo);
                        setTimeout(poll, 2000); // 2秒后再次检查
                    }
                } else {
                    if (onError) onError(data.message || '获取进度失败');
                }
            })
            .catch(error => {
                console.error('获取进度失败:', error);
                if (onError) onError('网络错误');
            });
        };

        poll();
    },

    // 停止分析
    stopAnalysis: function(sessionId) {
        window.TASApp.analysisStatus = 'stopped';
        // 如果有WebSocket连接，关闭它
        if (window.TASApp.wsConnection) {
            window.TASApp.wsConnection.close();
        }
    }
};

// 预设配置管理
const PresetManager = {
    presets: {
        'fast': {
            name: '快速分析',
            description: '适用于快速预览，参数较为宽松',
            n_components: 2,
            max_iter: 50,
            wavelength_range: [400, 700],
            delay_range: [0.1, 1000],
            language: 'zh'
        },
        'standard': {
            name: '标准分析',
            description: '平衡速度和精度，适用于大多数情况',
            n_components: 3,
            max_iter: 100,
            wavelength_range: [350, 750],
            delay_range: [0.1, 3000],
            language: 'zh'
        },
        'detailed': {
            name: '详细分析',
            description: '高精度分析，耗时较长但结果更准确',
            n_components: 4,
            max_iter: 200,
            wavelength_range: [300, 800],
            delay_range: [0.05, 5000],
            language: 'zh'
        },
        'custom': {
            name: '自定义配置',
            description: '手动设置所有参数',
            n_components: 3,
            max_iter: 100,
            wavelength_range: [400, 700],
            delay_range: [0.1, 1000],
            language: 'zh'
        }
    },

    // 应用预设
    applyPreset: function(presetKey) {
        const preset = this.presets[presetKey];
        if (!preset) return;

        // 更新表单字段
        const fields = {
            'n_components': preset.n_components,
            'max_iter': preset.max_iter,
            'wavelength_min': preset.wavelength_range[0],
            'wavelength_max': preset.wavelength_range[1],
            'delay_min': preset.delay_range[0],
            'delay_max': preset.delay_range[1],
            'language': preset.language
        };

        Object.entries(fields).forEach(([fieldName, value]) => {
            const field = document.querySelector(`[name="${fieldName}"]`);
            if (field) {
                field.value = value;
                // 触发change事件以更新任何依赖
                field.dispatchEvent(new Event('change'));
            }
        });

        // 更新预设选择器UI
        document.querySelectorAll('.preset-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        const selectedCard = document.querySelector(`[data-preset="${presetKey}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }

        Utils.showToast(`已应用${preset.name}配置`, 'success');
    },

    // 获取当前参数
    getCurrentParameters: function() {
        const form = document.getElementById('analysisForm');
        if (!form) return null;

        const formData = new FormData(form);
        return {
            n_components: parseInt(formData.get('n_components')) || 3,
            max_iter: parseInt(formData.get('max_iter')) || 100,
            wavelength_range: [
                parseFloat(formData.get('wavelength_min')) || 400,
                parseFloat(formData.get('wavelength_max')) || 700
            ],
            delay_range: [
                parseFloat(formData.get('delay_min')) || 0.1,
                parseFloat(formData.get('delay_max')) || 1000
            ],
            language: formData.get('language') || 'zh'
        };
    }
};

// 图表管理
const ChartManager = {
    // 创建进度图表
    createProgressChart: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        // 这里可以集成Chart.js或其他图表库
        // 暂时返回一个简单的进度条实现
        container.innerHTML = `
            <div class="progress mb-3">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 0%"></div>
            </div>
            <div class="text-center">
                <small class="text-muted">准备开始分析...</small>
            </div>
        `;

        return {
            updateProgress: function(percent, message) {
                const progressBar = container.querySelector('.progress-bar');
                const messageEl = container.querySelector('small');
                
                if (progressBar) {
                    progressBar.style.width = `${percent}%`;
                    progressBar.setAttribute('aria-valuenow', percent);
                }
                
                if (messageEl && message) {
                    messageEl.textContent = message;
                }
            }
        };
    }
};

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化Bootstrap组件
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 初始化预设配置选择器
    document.querySelectorAll('.preset-card').forEach(card => {
        card.addEventListener('click', function() {
            const presetKey = this.dataset.preset;
            if (presetKey) {
                PresetManager.applyPreset(presetKey);
            }
        });
    });

    // 初始化表单验证
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // 添加一些全局快捷键
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter 提交表单
        if (e.ctrlKey && e.key === 'Enter') {
            const submitBtn = document.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
        
        // Esc 关闭模态框
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const modal = bootstrap.Modal.getInstance(openModal);
                if (modal) modal.hide();
            }
        }
    });

    // 自动隐藏Flash消息
    const alerts = document.querySelectorAll('.alert[data-auto-dismiss]');
    alerts.forEach(alert => {
        const timeout = parseInt(alert.dataset.autoDismiss) || 5000;
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, timeout);
    });
});

// 导出到全局
window.Utils = Utils;
window.FileUpload = FileUpload;
window.AnalysisManager = AnalysisManager;
window.PresetManager = PresetManager;
window.ChartManager = ChartManager;
