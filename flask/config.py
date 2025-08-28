# TAS MCR-ALS 分析平台配置文件

import os
from datetime import timedelta

class Config:
    """基础配置类"""
    
    # 基本设置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'tas-mcr-als-dev-key-2024'
    
    # 文件上传设置
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'dat'}
    
    # 会话设置
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # 分析设置
    MAX_ANALYSIS_TIME = 3600  # 最大分析时间（秒）
    CLEANUP_INTERVAL = 86400  # 清理间隔（秒）
    
    # 语言设置
    LANGUAGES = {
        'zh': '中文',
        'en': 'English'
    }
    
    # 中文字体设置
    CHINESE_FONTS = [
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans',
        'Arial Unicode MS'
    ]

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    TESTING = False
    
    # 生产环境下使用更安全的密钥
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(32)

class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
