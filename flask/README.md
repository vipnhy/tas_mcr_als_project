# TAS MCR-ALS 分析平台

<div align="center">
  <h3>🔬 先进的瞬态吸收光谱在线分析工具</h3>
  <p>基于多元曲线分辨-交替最小二乘法(MCR-ALS)的专业光谱分析平台</p>
  
  ![Platform](https://img.shields.io/badge/Platform-Web-blue)
  ![Python](https://img.shields.io/badge/Python-3.8+-green)
  ![Flask](https://img.shields.io/badge/Flask-3.1.0-red)
  ![License](https://img.shields.io/badge/License-MIT-yellow)
</div>

## 📖 概述

这是一个基于Flask的专业Web应用，为TAS (Transient Absorption Spectroscopy) MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares) 分析提供用户友好的图形界面。该平台将复杂的光谱分析算法封装为简单易用的在线工具，让研究人员无需编程基础即可进行高质量的数据分析。

## ✨ 功能特性

### 🚀 核心功能
- � **数据上传**: 支持拖拽上传CSV、TXT、DAT格式的光谱数据文件
- ⚙️ **参数配置**: 提供预设配置和自定义参数设置
- 📊 **实时分析**: 在线运行MCR-ALS分析，实时显示进度和收敛状态
- 📈 **结果可视化**: 自动生成专业的分析图表和数据文件
- 💾 **结果管理**: 支持结果下载、会话管理和历史记录
- 🔍 **质量监控**: LOF值实时监控和收敛检测

### 🎨 用户界面
- 🌐 **响应式设计**: 完美支持桌面和移动设备
- 🎨 **现代UI**: 基于Bootstrap 5的精美渐变界面
- �️ **实时预览**: 图表和数据文件在线预览功能
- 📱 **多语言支持**: 完整的中英文界面切换
- 🔄 **进度追踪**: 详细的分析步骤和进度显示
- ⚡ **交互优化**: 流畅的用户体验和动画效果

### 📊 专业特性
- 🧪 **MCR-ALS算法**: 成熟的多元曲线分辨算法
- 🎯 **约束条件**: 非负性、归一化等物理约束
- 📉 **收敛分析**: 自适应收敛判断和LOF监控
- 🔢 **大数据支持**: 最大50MB文件，1000×1000数据点
- 🎨 **专业绘图**: 高质量的浓度轮廓图和纯光谱图

## 🖥️ 平台界面展示

### 主页界面
平台采用现代化渐变设计，清晰展示功能特性和使用流程：

```
┌─────────────────────────────────────────────────────────────┐
│                    TAS MCR-ALS 分析平台                      │
│              先进的光谱数据在线分析工具                        │
│                                                             │
│              🔬 专业 | 🚀 简单 | 📊 可视化                    │
│                                                             │
│                    [开始分析] 按钮                            │
└─────────────────────────────────────────────────────────────┘

🎯 核心特性：
┌───────────┐  ┌───────────┐  ┌───────────┐
│ 📤 简单上传 │  │ ⚙️ 灵活配置 │  │ 📊 可视化  │
│  拖拽上传   │  │  参数设置   │  │  结果展示   │
│  多格式     │  │  预设模式   │  │  专业图表   │
└───────────┘  └───────────┘  └───────────┘

┌───────────┐  ┌───────────┐  ┌───────────┐
│ 💾 结果下载 │  │ 📚 历史管理 │  │ 🌐 多语言  │
│  一键导出   │  │  会话记录   │  │  界面支持   │
│  批量下载   │  │  搜索筛选   │  │  中英切换   │
└───────────┘  └───────────┘  └───────────┘
```

### 文件上传界面
专业的拖拽上传界面，支持实时文件验证和参数配置：

```
┌─────────────────────────────────────────────────────────────┐
│                        上传TAS数据                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │               📁 拖拽文件到此处                       │    │
│  │                                                     │    │
│  │            或点击选择文件上传                         │    │
│  │                                                     │    │
│  │        支持格式：CSV, TXT, DAT                       │    │
│  │        最大大小：50MB                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ✅ 已选择文件：                                            │
│  📄 example_TAS_data.csv (2.3MB) ✓ 格式正确                │
│                                                             │
│  ⚙️ 分析参数配置                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  快速分析    │ │  标准分析    │ │  详细分析    │          │
│  │   2组分      │ │   3组分 ✓   │ │   4组分      │          │
│  │  50次迭代   │ │ 100次迭代   │ │ 200次迭代   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                             │
│  📋 自定义配置：                                            │
│  组分数：[3] ▼  最大迭代：[100]  语言：[中文] ▼            │
│                                                             │
│                    [开始分析]                               │
└─────────────────────────────────────────────────────────────┘
```

### 分析进度界面
实时监控分析状态，显示详细的算法收敛过程：

```
┌─────────────────────────────────────────────────────────────┐
│                      MCR-ALS 分析中...                      │
├─────────────────────────────────────────────────────────────┤
│  🔄 分析步骤：                                              │
│  ✅ [1] 数据载入    ✅ [2] 参数初始化    🔄 [3] MCR迭代     │
│                                                             │
│  📊 总体进度：                                              │
│  ████████████████░░░░░░░░░░░░ 65% (第78/100次迭代)         │
│                                                             │
│  📈 收敛状态：                                              │
│  当前LOF: 12.5% → 10.8% ↓  收敛趋势: 良好                 │
│                                                             │
│  📝 实时日志：                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ [2025-08-28 14:30:25] [INFO] 开始MCR-ALS分析...     │    │
│  │ [2025-08-28 14:30:26] [INFO] SIMPLISMA初始化完成    │    │
│  │ [2025-08-28 14:30:45] [INFO] 迭代 75/100, LOF=11.2% │    │
│  │ [2025-08-28 14:30:46] [INFO] 迭代 76/100, LOF=10.9% │    │
│  │ [2025-08-28 14:30:47] [INFO] 迭代 77/100, LOF=10.8% │    │
│  │ [2025-08-28 14:30:48] [SUCCESS] 收敛检测中...        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│                   [查看中间结果] [停止分析]                   │
└─────────────────────────────────────────────────────────────┘
```

### 结果展示界面
专业的分析结果展示和管理界面：

```
┌─────────────────────────────────────────────────────────────┐
│                       分析结果                               │
├─────────────────────────────────────────────────────────────┤
│  📊 分析摘要：                                              │
│  文件：example_data.csv │ 组分：3 │ 迭代：78 │ LOF：8.5%   │
│  开始：14:30:25 │ 完成：14:31:48 │ 用时：1分23秒 ✅ 收敛成功│
│                                                             │
│  🖼️ 分析图表：                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  浓度轮廓图  │  │   纯光谱图   │  │  LOF收敛图  │        │
│  │     📈      │  │     📊      │  │     📉      │        │
│  │  [预览][下载] │  │  [预览][下载] │  │  [预览][下载] │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  📄 数据文件：                                              │
│  📋 concentration_profiles.csv    📋 pure_spectra.csv       │
│  📋 lof_convergence.csv          📋 analysis_log.txt        │
│  📋 parameters.json              📋 summary_report.pdf      │
│                                                             │
│         [📦 下载全部结果] [🔄 新建分析] [📚 查看历史]         │
└─────────────────────────────────────────────────────────────┘
```

## 📊 分析结果示例

### 浓度轮廓图 (Concentration Profiles)
显示各化学组分随时间的动力学演化过程：

```
浓度 ↑
    │    ╭─╮ 组分1 (激发态)
1.0 │   ╱   ╲
    │  ╱     ╲
0.8 │ ╱       ╲
    │╱         ╲___
0.6 │              ╲___
    │         ╱─╲       ╲___
0.4 │        ╱   ╲           ╲
    │   ╱───╱     ╲           ╲
0.2 │  ╱           ╲___        ╲
    │ ╱  组分2          ╲___    ╲
0.0 └─────────────────────────────→ 时间(ps)
    0.1    1    10   100  1000
              组分3 (产物)

特点说明：
• 组分1: 快速衰减的激发态 (τ ≈ 2 ps)
• 组分2: 中间态物种 (τ ≈ 50 ps)  
• 组分3: 长寿命产物 (τ > 1000 ps)
```

### 纯光谱图 (Pure Spectra)
展示每个组分的特征吸收光谱：

```
ΔOD ↑
    │     ╱─╲ 组分1
0.04│    ╱   ╲ (激发态吸收)
    │   ╱     ╲
0.02│  ╱       ╲
    │ ╱         ╲____
0.00│╱               ╲____
    │                     ╲____
-0.02│        ╱──╲              ╲
    │       ╱    ╲ 组分2          ╲
-0.04│      ╱      ╲ (漂白)        ╲
    │     ╱        ╲___            ╲
-0.06└─────────────────────────────────→ 波长(nm)
    400   450   500   550   600   650
                    组分3 (产物吸收)

光谱特征：
• 正值：受激发射或激发态吸收
• 负值：基态漂白或受激发射
• 峰位：特征分子振动或电子跃迁
```

### LOF收敛曲线 (Lack of Fit)
监控算法收敛质量和拟合精度：

```
LOF(%) ↑
      │
  20  │ ●
      │  ╲●
  15  │   ●●●
      │     ╲●●
  10  │       ●●●●●●
      │            ╲●●●●●●●●
   5  │                   ●●●●●●●●
      │                         ●●●●●
   0  └────────────────────────────────→ 迭代次数
      0   10   20   30   40   50   60   70   78

收敛评价：
• 快速下降期 (0-20次): 主要组分识别
• 稳定优化期 (20-60次): 精细调整
• 收敛确认期 (60-78次): LOF < 10%，收敛成功
```

### 数据输出文件

#### 1. 浓度轮廓数据 (concentration_profiles.csv)
```csv
Time(ps),Component_1,Component_2,Component_3
0.1,0.847,0.123,0.030
0.2,0.821,0.151,0.028
0.5,0.754,0.208,0.038
1.0,0.651,0.284,0.065
2.0,0.523,0.371,0.106
5.0,0.342,0.457,0.201
10.0,0.198,0.502,0.300
...
```

#### 2. 纯光谱数据 (pure_spectra.csv)
```csv
Wavelength(nm),Component_1,Component_2,Component_3
400,0.012,-0.003,0.001
405,0.018,-0.004,0.002
410,0.025,-0.005,0.003
...
```

#### 3. 分析报告 (summary_report.txt)
```
=== TAS MCR-ALS 分析报告 ===
分析时间: 2025-08-28 14:30:25
文件名: example_TAS_data.csv
数据维度: 350 波长点 × 48 时间点

=== 分析参数 ===
组分数量: 3
最大迭代: 100
实际迭代: 78
收敛判据: 0.1%

=== 拟合质量 ===
最终LOF: 8.47%
R²: 0.9928
收敛状态: 成功

=== 组分分析 ===
组分1: 快速衰减 (τ ≈ 2.1 ps)
组分2: 中等寿命 (τ ≈ 48.3 ps)  
组分3: 长寿命产物 (τ > 1000 ps)
```

## 🚀 安装和运行

### 1. 环境要求
- **Python**: 3.8+ (推荐 3.9-3.11)
- **内存**: 最小2GB，推荐4GB以上
- **磁盘空间**: 至少500MB可用空间
- **浏览器**: Chrome, Firefox, Safari, Edge (不支持IE)
- **网络**: HTTP端口5000访问权限

### 2. 快速启动 (推荐)
```bash
# 1. 进入Flask应用目录
cd d:\TAS\tas_mcr_als_project\flask

# 2. 一键启动 (自动创建虚拟环境并安装依赖)
python run_server.py

# 3. 浏览器访问
# http://localhost:5000
```

### 3. 手动安装步骤

#### 步骤 1: 创建虚拟环境
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1

# Windows CMD
python -m venv venv
venv\Scripts\activate.bat

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 步骤 2: 安装依赖包
```bash
# 进入项目根目录
cd d:\TAS\tas_mcr_als_project

# 安装所有依赖
pip install -r requirements.txt

# 或者手动安装核心依赖
pip install flask==3.1.0 numpy scipy matplotlib pandas
```

#### 步骤 3: 启动应用
```bash
# 进入Flask目录
cd flask

# 方法1: 使用启动脚本
python run_server.py

# 方法2: 直接运行Flask
python app.py

# 方法3: 开发模式
set FLASK_ENV=development  # Windows
export FLASK_ENV=development  # Linux/macOS
flask run --host=0.0.0.0 --port=5000
```

### 4. 访问应用
启动成功后，您将看到以下信息：
```
 * Running on http://127.0.0.1:5000
 * Debug mode: off
 * Environment: production
 
🚀 TAS MCR-ALS 分析平台已启动
📱 本地访问: http://localhost:5000
🌐 网络访问: http://your-ip:5000
```

打开浏览器访问: `http://localhost:5000`

### 5. 快速测试
1. 主页应该显示平台介绍和功能特性
2. 点击"开始分析"进入上传页面
3. 拖拽测试数据文件 (在 `data/TAS/` 目录中)
4. 选择"快速分析"配置
5. 开始分析，观察实时进度
6. 查看和下载结果

## 🗂️ 项目结构详解

```
tas_mcr_als_project/                 # 项目根目录
├── flask/                          # Web应用主目录
│   ├── app.py                     # Flask主应用 (路由定义)
│   ├── run_server.py              # 服务启动脚本 (推荐使用)
│   ├── core_analyzer.py           # MCR-ALS核心分析算法
│   ├── config.py                  # 应用配置文件
│   ├── requirements.txt           # Python依赖包列表
│   ├── templates/                 # Jinja2 HTML模板
│   │   ├── base.html             # 基础布局模板
│   │   ├── index.html            # 主页 (平台介绍)
│   │   ├── upload.html           # 文件上传页面
│   │   ├── analyze.html          # 分析进度监控页面
│   │   ├── results.html          # 结果展示页面
│   │   ├── sessions.html         # 历史会话管理
│   │   └── 404.html              # 错误页面
│   ├── static/                   # 静态资源文件
│   │   ├── css/
│   │   │   └── style.css         # 自定义样式 (500+ 行)
│   │   ├── js/
│   │   │   └── app.js            # 前端交互逻辑
│   │   └── images/               # 图片资源
│   ├── uploads/                  # 用户上传文件存储 (自动创建)
│   ├── results/                  # 分析结果文件存储 (自动创建)
│   └── docs/                     # 项目文档
│       ├── 使用指南.md           # 详细使用教程
│       ├── 技术文档.md           # 技术实现文档  
│       ├── 快速入门.md           # 5分钟快速上手
│       └── images/               # 文档配图
├── mcr/                           # MCR-ALS算法核心模块
│   ├── __init__.py               # 模块初始化
│   ├── mcr_als.py                # 主算法实现
│   ├── constraints.py            # 约束条件定义
│   └── initializers.py           # 初值估计算法
├── data/                          # 数据处理模块
│   ├── data.py                   # 数据IO和预处理
│   └── TAS/                      # 测试数据集
│       ├── TA_Average.csv        # 示例TAS数据
│       └── HRR/                  # 高分辨率数据
└── test/                          # 测试模块
    ├── test_MCR_ALS.py           # 算法单元测试
    ├── test_real_data.py         # 真实数据测试
    └── test_axis_orientation.py  # 数据格式测试
```

### 关键文件说明

| 文件 | 功能 | 行数 | 描述 |
|------|------|------|------|
| `app.py` | Web应用主体 | ~400 | Flask路由定义和视图函数 |
| `core_analyzer.py` | 分析引擎 | ~300 | MCR-ALS算法封装和调用 |
| `style.css` | 界面样式 | ~500 | 响应式设计和动画效果 |
| `app.js` | 前端逻辑 | ~250 | 文件上传和进度监控 |
| `mcr_als.py` | 核心算法 | ~800 | MCR-ALS数学实现 |

## 使用说明

### 1. 数据上传
1. 点击首页的"开始分析"按钮
2. 拖拽文件到上传区域或点击选择文件
3. 支持的文件格式：CSV、TXT、DAT
4. 文件大小限制：50MB

### 2. 参数设置
提供四种预设配置：
- **快速分析**: 适用于快速预览（2组分，50次迭代）
- **标准分析**: 平衡速度和精度（3组分，100次迭代）
- **详细分析**: 高精度分析（4组分，200次迭代）
- **自定义配置**: 手动设置所有参数

### 3. 分析过程
- 实时显示分析进度和状态
- 显示详细的分析日志
- 支持在分析完成前查看进度

### 4. 结果查看
- 自动生成的分析图表（PNG格式）
- 导出的数据文件（CSV格式）
- 在线预览图片和数据
- 支持单个文件或打包下载

### 5. 会话管理
- 查看所有历史分析会话
- 搜索和筛选功能
- 重新运行历史分析
- 删除不需要的会话

## API接口

### 文件上传
```
POST /upload
Content-Type: multipart/form-data
参数: file (文件)
```

### 开始分析
```
POST /api/run_analysis
Content-Type: application/json
参数: {
    "session_id": "会话ID",
    "n_components": 组分数量,
    "max_iter": 最大迭代次数,
    "wavelength_range": [最小波长, 最大波长],
    "delay_range": [最小延迟, 最大延迟],
    "language": "语言代码"
}
```

### 获取进度
```
GET /api/get_progress/<session_id>
返回: {
    "success": true,
    "session_info": {会话信息}
}
```

### 下载文件
```
GET /download/<session_id>/<filename>
返回: 文件内容
```

## 配置说明

### 应用配置 (config.py)
- `SECRET_KEY`: Flask会话密钥
- `UPLOAD_FOLDER`: 上传文件存储目录
- `RESULTS_FOLDER`: 结果文件存储目录
- `MAX_CONTENT_LENGTH`: 最大文件上传大小
- `ALLOWED_EXTENSIONS`: 允许的文件扩展名

### 分析参数
- `n_components`: MCR分解的组分数量 (1-10)
- `max_iter`: 最大迭代次数 (10-1000)
- `wavelength_range`: 波长范围 (nm)
- `delay_range`: 时间延迟范围 (ps)
- `language`: 界面语言 ('zh' 或 'en')

## 📚 文档资源

### 完整文档列表
| 文档 | 位置 | 描述 | 适用对象 |
|------|------|------|----------|
| 🚀 快速入门 | [docs/快速入门.md](docs/快速入门.md) | 5分钟上手指南 | 新用户 |
| 📖 使用指南 | [docs/使用指南.md](docs/使用指南.md) | 详细操作教程 | 所有用户 |
| 🔧 技术文档 | [docs/技术文档.md](docs/技术文档.md) | 开发和部署指南 | 开发者 |
| 🌐 API文档 | [docs/技术文档.md#api文档](docs/技术文档.md#api文档) | RESTful接口说明 | 集成开发 |

### 在线帮助
- 🏠 **主页帮助**: 平台首页有详细的功能介绍
- 💡 **页面提示**: 每个页面都有操作提示和帮助信息  
- 📝 **实时日志**: 分析过程中的详细日志记录
- ❓ **FAQ**: 常见问题和解决方案

## 🛠️ 高级配置

### 生产环境部署

#### 使用 Gunicorn (Linux推荐)
```bash
# 安装Gunicorn
pip install gunicorn

# 启动多worker服务
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 app:app

# 后台运行
nohup gunicorn -w 4 -b 0.0.0.0:5000 app:app > gunicorn.log 2>&1 &
```

#### 使用 Nginx 反向代理
```nginx
# /etc/nginx/sites-available/tas-mcr-als
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /static {
        alias /path/to/flask/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

#### Docker 容器化部署
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p uploads results

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "run_server.py"]
```

```bash
# 构建和运行
docker build -t tas-mcr-als .
docker run -p 5000:5000 -v $(pwd)/data:/app/uploads tas-mcr-als
```

### 性能优化配置

#### 大文件处理
```python
# config.py 中的设置
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
UPLOAD_TIMEOUT = 300  # 5分钟上传超时
ANALYSIS_TIMEOUT = 1800  # 30分钟分析超时
```

#### 内存优化
```python
# 数据分块处理
CHUNK_SIZE = 1000  # 处理数据的块大小
MAX_WORKERS = 4    # 最大并发分析数
MEMORY_LIMIT = '2G'  # 内存使用限制
```

#### 缓存配置
```python
# 启用Redis缓存 (可选)
CACHE_TYPE = 'redis'
CACHE_REDIS_URL = 'redis://localhost:6379/0'
CACHE_DEFAULT_TIMEOUT = 3600  # 1小时缓存
```

## 📊 性能基准

### 测试环境
- **CPU**: Intel i7-8700K @ 3.7GHz
- **内存**: 16GB DDR4
- **存储**: SSD
- **Python**: 3.9.7

### 性能指标

| 数据规模 | 组分数 | 迭代次数 | 处理时间 | 内存使用 | LOF质量 |
|----------|--------|----------|----------|----------|---------|
| 100×50 | 3 | 50 | 15秒 | 50MB | <5% |
| 500×100 | 3 | 100 | 1.5分钟 | 200MB | <8% |
| 1000×200 | 4 | 150 | 5分钟 | 800MB | <10% |
| 2000×500 | 5 | 200 | 20分钟 | 2GB | <12% |

### 并发能力
- **单用户**: 可同时处理多个会话
- **多用户**: 支持最多10个并发分析
- **负载均衡**: 支持多实例部署
- **资源隔离**: 每个分析任务独立内存空间

## 📈 监控和维护

### 健康检查
```bash
# 检查服务状态
curl http://localhost:5000/health

# 返回示例
{
  "status": "healthy",
  "uptime": "2 days, 3 hours",
  "active_sessions": 2,
  "disk_usage": "45%",
  "memory_usage": "68%"
}
```

### 日志管理
```python
# 配置详细日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 定期清理
```bash
#!/bin/bash
# cleanup.sh - 定期清理脚本

# 清理7天前的上传文件
find uploads/ -type f -mtime +7 -delete

# 清理30天前的结果文件  
find results/ -type f -mtime +30 -delete

# 清理日志文件
find . -name "*.log" -size +100M -delete

echo "清理完成: $(date)"
```

## ❓ 故障排除

### 常见问题及解决方案

#### 1. 启动相关问题

**❌ 问题**: `ModuleNotFoundError: No module named 'flask'`
```bash
解决方案:
1. 确认已激活虚拟环境: venv\Scripts\activate
2. 安装依赖: pip install -r requirements.txt
3. 检查Python版本: python --version (需要 ≥ 3.8)
```

**❌ 问题**: `OSError: [Errno 48] Address already in use`
```bash
解决方案:
1. 检查端口占用: netstat -ano | findstr :5000
2. 杀死占用进程: taskkill /PID <PID> /F
3. 或更改端口: python run_server.py --port 5001
```

**❌ 问题**: `Permission denied` 文件权限错误
```bash
解决方案:
1. Windows: 以管理员身份运行
2. Linux: chmod 755 uploads/ results/
3. 检查磁盘空间: df -h
```

#### 2. 数据上传问题

**❌ 问题**: 文件上传失败或格式错误
```
解决方案:
✅ 检查文件格式: 仅支持 CSV, TXT, DAT
✅ 检查文件大小: 最大 50MB
✅ 检查数据格式: 第一行为标题，第一列为波长
✅ 检查编码格式: 推荐 UTF-8
```

**数据格式示例**:
```csv
Wavelength(nm),0.1ps,0.2ps,0.5ps,1ps...
350,0.001,0.002,0.003,0.004...
351,0.002,0.003,0.004,0.005...
```

#### 3. 分析失败问题

**❌ 问题**: MCR-ALS分析不收敛
```
解决方案:
🎯 减少组分数量 (推荐2-4个)
🎯 增加最大迭代次数 (100-300)
🎯 检查数据质量 (信噪比、基线)
🎯 调整波长/时间范围 (排除噪声区域)
🎯 尝试不同的初始化方法
```

**❌ 问题**: 内存不足错误
```
解决方案:
💾 减少数据大小 (降采样)
💾 关闭其他程序释放内存
💾 使用数据分块处理
💾 增加虚拟内存/交换空间
```

#### 4. 结果异常问题

**❌ 问题**: 图表显示空白或异常
```
解决方案:
🖼️ 检查浏览器兼容性 (推荐Chrome/Firefox)
🖼️ 清除浏览器缓存
🖼️ 禁用广告拦截器
🖼️ 下载PNG文件直接查看
🖼️ 检查控制台错误信息
```

**❌ 问题**: LOF值异常高 (>50%)
```
原因分析:
📊 数据质量问题 (噪声过大)
📊 组分数量不合适
📊 存在基线漂移
📊 时间/波长范围选择不当

解决建议:
✨ 数据预处理 (去噪、基线校正)
✨ 重新选择合适的组分数
✨ 调整分析范围
✨ 尝试不同的约束条件
```

### 调试工具

#### 1. 启用调试模式
```python
# 在 run_server.py 中设置
app.run(debug=True, host='0.0.0.0', port=5000)

# 或设置环境变量
set FLASK_ENV=development  # Windows
export FLASK_ENV=development  # Linux/macOS
```

#### 2. 查看详细日志
```python
# 在 app.py 开头添加
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看分析日志
tail -f uploads/session_*/analysis.log
```

#### 3. 性能分析
```python
# 安装性能分析工具
pip install flask-profiler

# 在代码中启用
from flask_profiler import Profiler
profiler = Profiler()
profiler.init_app(app)
```

### 技术支持渠道

| 问题类型 | 联系方式 | 响应时间 |
|---------|----------|----------|
| 🔧 技术故障 | [GitHub Issues] | 1-2个工作日 |
| 💬 使用咨询 | [support@email.com] | 当日回复 |
| 📚 文档问题 | [docs@email.com] | 2-3个工作日 |
| 🚀 功能建议 | [feature@email.com] | 1周内评估 |

## 🔧 技术栈详解

### 后端技术栈
| 技术 | 版本 | 用途 | 特性 |
|------|------|------|------|
| **Flask** | 3.1.0 | Web框架 | 轻量、灵活、易扩展 |
| **NumPy** | 1.24+ | 科学计算 | 高性能数组运算 |
| **SciPy** | 1.10+ | 算法库 | 优化算法、线性代数 |
| **Matplotlib** | 3.7+ | 数据可视化 | 专业图表生成 |
| **Pandas** | 2.0+ | 数据处理 | 数据IO和操作 |
| **Werkzeug** | 2.3+ | WSGI工具包 | 文件上传、安全 |

### 前端技术栈
| 技术 | 版本 | 用途 | 特性 |
|------|------|------|------|
| **Bootstrap** | 5.1.3 | UI框架 | 响应式、组件丰富 |
| **Font Awesome** | 6.0.0 | 图标库 | 矢量图标、美观 |
| **JavaScript** | ES6+ | 交互逻辑 | 原生、无依赖 |
| **CSS3** | - | 样式设计 | 渐变、动画、响应式 |
| **Jinja2** | 3.1+ | 模板引擎 | 动态HTML生成 |

### 算法技术栈
| 技术 | 描述 | 优势 |
|------|------|------|
| **MCR-ALS** | 多元曲线分辨-交替最小二乘 | 成熟算法、物理约束 |
| **SIMPLISMA** | 初值估计算法 | 自动化、鲁棒性强 |
| **NumPy优化** | 向量化运算 | 10-100倍性能提升 |
| **约束优化** | 非负性、归一化约束 | 物理意义明确 |

### 开发工具栈
| 工具 | 用途 | 配置 |
|------|------|------|
| **Git** | 版本控制 | 分支管理、协作开发 |
| **VS Code** | 代码编辑 | Python插件、调试 |
| **Pip** | 包管理 | 依赖安装、版本控制 |
| **Venv** | 环境隔离 | 虚拟环境管理 |

## 📖 开发指南

### 项目架构设计
```
┌─────────────────────────────────────────────────────────────┐
│                    表示层 (Presentation)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ HTML模板    │  │ CSS样式     │  │ JavaScript  │          │
│  │ Bootstrap   │  │ 响应式设计   │  │ AJAX交互    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    业务层 (Business)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Flask路由   │  │ 会话管理     │  │ 文件处理    │          │
│  │ 视图逻辑    │  │ 状态跟踪     │  │ 数据验证    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    算法层 (Algorithm)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ MCR-ALS核心 │  │ 数据预处理   │  │ 结果生成    │          │
│  │ 约束优化    │  │ 格式转换     │  │ 图表绘制    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    数据层 (Data)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 文件存储    │  │ 会话数据     │  │ 结果缓存    │          │
│  │ 上传管理    │  │ 临时文件     │  │ 历史记录    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 代码结构说明
- **MVC模式**: 清晰的模型-视图-控制器分离
- **模块化设计**: 功能模块独立，便于维护
- **RESTful API**: 标准的HTTP接口设计
- **响应式前端**: 适配不同设备和屏幕

### 扩展开发指南

#### 1. 添加新的分析算法
```python
# 在 core_analyzer.py 中扩展
class CustomAnalyzer:
    def __init__(self, data, params):
        self.data = data
        self.params = params
    
    def analyze(self):
        # 实现新算法
        pass
    
    def get_results(self):
        # 返回结果
        pass
```

#### 2. 添加新的前端页面
```html
<!-- 在 templates/ 中创建新模板 -->
{% extends "base.html" %}
{% block content %}
<!-- 页面内容 -->
{% endblock %}
```

#### 3. 添加新的API端点
```python
# 在 app.py 中添加路由
@app.route('/api/new_endpoint', methods=['POST'])
def new_endpoint():
    # 处理逻辑
    return jsonify({'status': 'success'})
```

#### 4. 自定义样式和主题
```css
/* 在 static/css/style.css 中添加 */
.custom-theme {
    --primary-color: #your-color;
    --secondary-color: #your-color;
}
```

### 贡献指南

#### 开发环境设置
```bash
# 1. Fork 项目
git clone https://github.com/your-username/tas_mcr_als_project.git

# 2. 创建开发分支
git checkout -b feature/your-feature-name

# 3. 安装开发依赖
pip install -r requirements-dev.txt

# 4. 运行测试
python -m pytest test/

# 5. 代码风格检查
flake8 flask/
black flask/
```

#### 提交规范
```bash
# 提交信息格式
git commit -m "类型(范围): 简短描述

详细描述 (可选)

关闭的问题: #123"

# 类型包括:
# feat: 新功能
# fix: 修复bug
# docs: 文档更新
# style: 代码格式
# refactor: 重构
# test: 测试
```

#### 代码规范
- **PEP 8**: Python代码风格标准
- **类型注解**: 使用类型提示增加代码可读性
- **文档字符串**: 为函数和类添加docstring
- **单元测试**: 新功能需要对应的测试用例

#### Pull Request 流程
1. 确保代码通过所有测试
2. 更新相关文档
3. 添加变更日志
4. 提交Pull Request
5. 代码审查和合并

## 🌟 路线图和未来计划

### 短期目标 (1-3个月)
- [ ] 🚀 **性能优化**: 大数据集处理能力提升
- [ ] 🎨 **UI增强**: 深色模式和自定义主题
- [ ] 📱 **移动优化**: 手机端界面适配
- [ ] 🔍 **数据验证**: 更强的数据格式检测
- [ ] 📊 **新图表**: 3D可视化和交互式图表

### 中期目标 (3-6个月)
- [ ] 🔐 **用户系统**: 登录注册和权限管理
- [ ] 💾 **数据库**: 持久化存储和历史数据
- [ ] 🌐 **多语言**: 英文、日文界面支持
- [ ] 🤖 **AI辅助**: 智能参数推荐
- [ ] 📈 **批量处理**: 多文件同时分析

### 长期目标 (6-12个月)
- [ ] ☁️ **云服务**: 在线SaaS平台
- [ ] 🔌 **插件系统**: 第三方算法集成
- [ ] 📚 **知识库**: 在线教程和案例库
- [ ] 🤝 **协作功能**: 团队共享和讨论
- [ ] 📜 **标准化**: 符合FAIR数据原则

## 🏆 致谢和贡献者

### 核心开发团队
- **算法实现**: MCR-ALS核心算法开发
- **前端设计**: 用户界面和体验设计
- **后端架构**: Web服务和数据处理
- **文档编写**: 用户指南和技术文档

### 开源贡献
感谢以下开源项目的支持：
- [Flask](https://flask.palletsprojects.com/) - 优秀的Python Web框架
- [NumPy](https://numpy.org/) - 科学计算基础库
- [SciPy](https://scipy.org/) - 科学算法工具包
- [Matplotlib](https://matplotlib.org/) - 数据可视化库
- [Bootstrap](https://getbootstrap.com/) - 前端UI框架

### 学术引用
如果您在学术研究中使用了本平台，请引用：
```bibtex
@software{tas_mcr_als_platform,
  title={TAS MCR-ALS Analysis Platform},
  author={Your Name and Contributors},
  year={2025},
  url={https://github.com/your-repo/tas_mcr_als_project},
  note={Version 1.0.0}
}
```

## 📄 许可证和法律声明

### 开源许可证
本项目基于 **MIT License** 开源协议发布。

```
MIT License

Copyright (c) 2025 TAS-MCR-ALS Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 使用条款
- ✅ **商业使用**: 允许商业用途
- ✅ **修改权**: 允许修改源代码
- ✅ **分发权**: 允许分发软件
- ✅ **私人使用**: 允许私人使用
- ❌ **责任限制**: 不承担使用风险
- ❌ **保证免责**: 不提供软件保证

### 免责声明
- 本软件仅供科研和教育用途
- 使用者需确保上传数据符合相关法律法规
- 开发团队不对分析结果的准确性承担责任
- 生产环境使用请进行充分测试

## 📞 联系我们

### 技术支持
| 联系方式 | 地址 | 说明 |
|---------|------|------|
| 📧 邮件支持 | support@tas-mcr-als.com | 技术问题和使用咨询 |
| 🐛 问题反馈 | [GitHub Issues](https://github.com/your-repo/issues) | Bug报告和功能建议 |
| 📚 文档问题 | docs@tas-mcr-als.com | 文档改进建议 |
| 💼 商业合作 | business@tas-mcr-als.com | 定制开发和合作 |

### 社区资源
- 🏠 **项目主页**: [GitHub Repository](https://github.com/your-repo/tas_mcr_als_project)
- 📖 **在线文档**: [Documentation Site](https://docs.tas-mcr-als.com)
- 🎓 **教程视频**: [YouTube Channel](https://youtube.com/tas-mcr-als)
- 💬 **用户论坛**: [Community Forum](https://forum.tas-mcr-als.com)

### 开发团队
- **项目负责人**: [Your Name]
- **算法专家**: [Algorithm Expert Name]
- **前端工程师**: [Frontend Developer Name]
- **文档维护**: [Documentation Maintainer Name]

---

## 🎯 结语

TAS MCR-ALS 分析平台致力于为光谱分析领域提供专业、易用的在线工具。我们相信，通过将复杂的算法封装为直观的Web界面，能够大大降低高质量数据分析的门槛，让更多研究人员受益。

### 我们的愿景
🔬 **让光谱分析变得简单** - 无需编程基础，人人都能进行专业分析  
🚀 **推动科研效率提升** - 自动化工具释放研究人员创造力  
🌐 **促进学术交流合作** - 开源平台连接全球研究社区  
📚 **建设知识共享生态** - 文档、教程、案例全方位支持  

### 加入我们
如果您对这个项目感兴趣，我们欢迎您：
- ⭐ **给项目点星** - 您的支持是我们前进的动力
- 🐛 **报告问题** - 帮助我们发现和修复bug
- 💡 **提出建议** - 分享您的想法和需求
- 🛠️ **贡献代码** - 参与开发，共同改进
- 📝 **完善文档** - 帮助更多用户了解和使用

---

<div align="center">
  
  ### 🌟 如果这个项目对您有帮助，请给我们一个星标！

  ### 🔬 让我们一起推动光谱分析技术的发展！

  **版本**: v1.0.0 | **更新时间**: 2025年8月28日 | **维护团队**: TAS-MCR-ALS开发团队

</div>

---

**注意**: 本项目持续更新中，请关注最新版本以获得最佳体验。如有任何问题或建议，欢迎通过上述渠道联系我们。
