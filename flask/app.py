#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py - TAS MCR-ALS Flask Web应用

提供用户友好的网页界面，用于上传TAS数据并进行MCR-ALS分析
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json
import uuid
from datetime import datetime
import zipfile
import io

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core_analyzer import TASCoreAnalyzer, analyze_tas_data

app = Flask(__name__)
app.secret_key = 'tas_mcr_als_secret_key_2025'  # 更改为更安全的密钥

# 配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'csv', 'txt', 'dat'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 存储活跃的分析会话
active_sessions = {}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """文件上传页面"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            flash('请选择文件')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('请选择文件')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # 生成安全的文件名
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = str(uuid.uuid4())[:8]
            safe_filename = f"{timestamp}_{session_id}_{filename}"
            
            # 保存文件
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(file_path)
            
            # 获取表单参数
            file_type = request.form.get('file_type', 'handle')
            wavelength_min = float(request.form.get('wavelength_min', 400))
            wavelength_max = float(request.form.get('wavelength_max', 800))
            delay_min = float(request.form.get('delay_min', 0))
            delay_max = float(request.form.get('delay_max', 10))
            n_components = int(request.form.get('n_components', 3))
            max_iter = int(request.form.get('max_iter', 200))
            language = request.form.get('language', 'chinese')
            
            # 创建分析会话
            session_info = {
                'session_id': session_id,
                'filename': filename,
                'file_path': file_path,
                'upload_time': datetime.now().isoformat(),
                'parameters': {
                    'file_type': file_type,
                    'wavelength_range': [wavelength_min, wavelength_max],
                    'delay_range': [delay_min, delay_max],
                    'n_components': n_components,
                    'max_iter': max_iter,
                    'language': language
                },
                'status': 'uploaded'
            }
            
            active_sessions[session_id] = session_info
            
            flash(f'文件 "{filename}" 上传成功！')
            return redirect(url_for('analyze', session_id=session_id))
        else:
            flash('不支持的文件格式。请上传 CSV, TXT 或 DAT 文件。')
    
    return render_template('upload.html')


@app.route('/analyze/<session_id>')
def analyze(session_id):
    """分析页面"""
    if session_id not in active_sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_info = active_sessions[session_id]
    return render_template('analyze.html', session_info=session_info)


@app.route('/api/run_analysis', methods=['POST'])
def api_run_analysis():
    """API: 运行MCR-ALS分析"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'message': '会话不存在或已过期'
            })
        
        session_info = active_sessions[session_id]
        params = session_info['parameters']
        
        # 更新状态
        session_info['status'] = 'analyzing'
        
        # 创建分析器
        analyzer = TASCoreAnalyzer(language=params['language'])
        
        # 执行分析
        results = {
            'session_id': session_id,
            'steps': []
        }
        
        # 1. 加载数据
        load_result = analyzer.load_data(
            session_info['file_path'],
            params['file_type'],
            tuple(params['wavelength_range']),
            tuple(params['delay_range'])
        )
        results['steps'].append({'step': 'data_loading', 'result': load_result})
        
        if not load_result['success']:
            session_info['status'] = 'error'
            return jsonify({
                'success': False,
                'message': load_result['message'],
                'results': results
            })
        
        # 2. 运行分析
        analysis_result = analyzer.run_analysis(
            params['n_components'],
            params['max_iter']
        )
        results['steps'].append({'step': 'analysis', 'result': analysis_result})
        
        if not analysis_result['success']:
            session_info['status'] = 'error'
            return jsonify({
                'success': False,
                'message': analysis_result['message'],
                'results': results
            })
        
        # 3. 生成图表
        session_results_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        plot_result = analyzer.generate_plots(session_results_dir)
        results['steps'].append({'step': 'plotting', 'result': plot_result})
        
        # 4. 保存结果
        save_result = analyzer.save_results(session_results_dir)
        results['steps'].append({'step': 'saving', 'result': save_result})
        
        # 5. 获取摘要
        summary_result = analyzer.get_analysis_summary()
        results['summary'] = summary_result
        
        # 更新会话信息
        session_info['status'] = 'completed'
        session_info['analysis_id'] = analyzer.analysis_id
        session_info['results_dir'] = session_results_dir
        session_info['completion_time'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': '分析完成',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'分析过程中发生错误: {str(e)}'
        })


@app.route('/api/get_progress/<session_id>')
def api_get_progress(session_id):
    """API: 获取分析进度"""
    if session_id not in active_sessions:
        return jsonify({
            'success': False,
            'message': '会话不存在'
        })
    
    session_info = active_sessions[session_id]
    return jsonify({
        'success': True,
        'status': session_info['status'],
        'session_info': session_info
    })


@app.route('/results/<session_id>')
def results(session_id):
    """结果页面"""
    if session_id not in active_sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_info = active_sessions[session_id]
    
    if session_info['status'] != 'completed':
        flash('分析尚未完成')
        return redirect(url_for('analyze', session_id=session_id))
    
    # 获取结果文件列表
    results_dir = session_info.get('results_dir', '')
    result_files = []
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            if os.path.isfile(file_path):
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'is_image': filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')),
                    'is_data': filename.lower().endswith(('.csv', '.json'))
                }
                result_files.append(file_info)
    
    return render_template('results.html', 
                         session_info=session_info, 
                         result_files=result_files)


@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """下载结果文件"""
    if session_id not in active_sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_info = active_sessions[session_id]
    results_dir = session_info.get('results_dir', '')
    
    file_path = os.path.join(results_dir, secure_filename(filename))
    
    if not os.path.exists(file_path):
        flash('文件不存在')
        return redirect(url_for('results', session_id=session_id))
    
    return send_file(file_path, as_attachment=True)


@app.route('/download_all/<session_id>')
def download_all_results(session_id):
    """下载所有结果文件（打包为ZIP）"""
    if session_id not in active_sessions:
        flash('会话不存在或已过期')
        return redirect(url_for('index'))
    
    session_info = active_sessions[session_id]
    results_dir = session_info.get('results_dir', '')
    
    if not os.path.exists(results_dir):
        flash('结果目录不存在')
        return redirect(url_for('results', session_id=session_id))
    
    # 创建内存中的ZIP文件
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, results_dir)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    # 生成下载文件名
    download_filename = f"TAS_MCR_ALS_Results_{session_id}.zip"
    
    return send_file(
        io.BytesIO(memory_file.read()),
        mimetype='application/zip',
        as_attachment=True,
        download_name=download_filename
    )


@app.route('/api/sessions')
def api_list_sessions():
    """API: 获取所有会话列表"""
    sessions_list = []
    for session_id, session_info in active_sessions.items():
        sessions_list.append({
            'session_id': session_id,
            'filename': session_info['filename'],
            'upload_time': session_info['upload_time'],
            'status': session_info['status'],
            'parameters': session_info['parameters']
        })
    
    return jsonify({
        'success': True,
        'sessions': sessions_list
    })


@app.route('/sessions')
def list_sessions():
    """会话管理页面"""
    return render_template('sessions.html')


@app.route('/api/delete_session/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """API: 删除会话"""
    if session_id not in active_sessions:
        return jsonify({
            'success': False,
            'message': '会话不存在'
        })
    
    try:
        session_info = active_sessions[session_id]
        
        # 删除上传的文件
        if os.path.exists(session_info['file_path']):
            os.remove(session_info['file_path'])
        
        # 删除结果目录
        results_dir = session_info.get('results_dir', '')
        if results_dir and os.path.exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        
        # 从活跃会话中移除
        del active_sessions[session_id]
        
        return jsonify({
            'success': True,
            'message': '会话已删除'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除会话时发生错误: {str(e)}'
        })


@app.errorhandler(413)
def too_large(e):
    """文件过大错误处理"""
    flash('文件过大，请上传小于50MB的文件')
    return redirect(url_for('upload_file')), 413


@app.errorhandler(404)
def not_found(e):
    """404错误处理"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """500错误处理"""
    return render_template('500.html'), 500


if __name__ == '__main__':
    # 开发模式
    app.run(debug=True, host='0.0.0.0', port=5000)
