# 🚀 FastInput - 智能输入法系统

> 基于 PyTorch + RNN 的中文智能输入预测系统，支持根据用户输入上下文预测下一个最可能的词语。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

**FastInput** 是一个轻量级的中文智能输入法预测系统，基于深度学习技术实现。系统通过学习大量中文对话语料，能够根据用户已输入的文本序列，智能预测下一个最可能输入的词语，提升输入效率。

## 📋 项目目录

fastinput/
├── 📄 README.md # 项目说明文档
├── 📄 pyproject.toml # 项目依赖配置
├── 📄 uv.lock # 依赖锁定文件
├── 📁 data/
│ ├── 📁 raw/ # 原始数据目录
│ │ └── synthesized_.jsonl
│ └── 📁 processed/ # 预处理后数据
│ ├── train.jsonl
│ └── test.jsonl
├── 📁 models/ # 模型文件目录
│ ├── vocab.txt # 词汇表
│ └── best.pth # 训练好的模型权重
├── 📁 logs/ # TensorBoard训练日志
├── 📁 src/ # 核心源代码
│ ├── config.py # 全局配置参数
│ ├── dataset.py # 数据集加载模块
│ ├── model.py # 模型定义
│ ├── train.py # 训练入口
│ ├── predict.py # 预测入口
│ └── process.py # 数据预处理脚本
└── 📁 test/ # 测试用例目录

### 核心流程

原始语料 → 分词预处理 → 序列构建 → RNN模型训练 → 实时预测

---

## ✨ 核心特性

- 🔤 **中文分词支持**：集成 `jieba` 分词，精准处理中文文本
- 🧠 **RNN序列建模**：采用循环神经网络捕捉文本上下文依赖关系
- ⚡ **高效预测**：支持 Top-K 候选词推荐，默认返回概率最高的5个词
- 📊 **训练可视化**：集成 TensorBoard，实时监控训练损失曲线
- 🔄 **模块化设计**：数据预处理、模型训练、预测推理解耦，便于扩展
- 🗃️ **自动词表构建**：从训练数据自动构建词汇表，支持未知词处理

---

## 🏗️ 技术架构

┌─────────────────────────────────────┐
│ 应用层 │
│ ┌─────────┐ ┌─────────┐ │
│ │ 交互式预测│ │ 批量预测 │ │
│ └─────────┘ └─────────┘ │
└─────────────────────────────────────┘
│
┌─────────────────────────────────────┐
│ 模型层 │
│ ┌─────────────────────────┐ │
│ │ FastInputModel (RNN) │ │
│ │ • Embedding 层 │ │
│ │ • RNN 隐藏层 │ │
│ │ • Linear 输出层 │ │
│ └─────────────────────────┘ │
└─────────────────────────────────────┘
│
┌─────────────────────────────────────┐
│ 数据层 │
│ ┌─────────┐ ┌─────────┐ │
│ │数据处理 │ │ DataLoader│ │
│ │• jieba分词│ │• 批量加载│ │
│ │• 序列构建│ │• 自动打散│ │
│ └─────────┘ └─────────┘ │
└─────────────────────────────────────┘

### 模型结构详情

| 组件        | 参数                      | 说明            |
|-----------|-------------------------|---------------|
| Embedding | `vocab_size × 128`      | 词向量嵌入层        |
| RNN       | `input=128, hidden=256` | 循环神经网络，捕捉序列特征 |
| Linear    | `256 → vocab_size`      | 输出层，映射到词表概率分布 |
| Loss      | CrossEntropyLoss        | 多分类交叉熵损失      |
| Optimizer | Adam (lr=1e-3)          | 自适应学习率优化器     |

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 克隆项目
git clone https://github.com/dezhizhang/fastinput.git
cd fastinput

# 安装依赖（推荐使用 uv 包管理器）
uv add
# 或使用 pip
pip install torch jieba pandas scikit-learn tensorboard
```

### 2️⃣ 数据准备

将原始对话数据放置于 data/raw/synthesized_.jsonl，格式如下：

```bash
{"dialog": ["用户：你好", "助手：您好，有什么可以帮您"]}
{"dialog": ["用户：今天天气", "助手：今天天气晴朗"]}
```

### 3️⃣ 数据预处理

```bash
cd src
python process.py
```

执行后将自动生成：

- models/vocab.txt：词汇表文件
- data/processed/train.jsonl：训练数据
- data/processed/test.jsonl：测试数据

### 4️⃣ 模型训练

```bash
python train.py
```

训练日志和模型将保存至：

- models/best.pth：最佳模型权重
- logs/：TensorBoard 日志目录

### 5️⃣ 启动预测

```bash
python predict.py
```

## 📖 使用指南

```python
from predict import predict
import torch
from model import FastInputModel
import config

# 加载资源
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
    vocab_list = [line.strip() for line in f.readlines()]
word2index = {w: i for i, w in enumerate(vocab_list)}
index2word = {i: w for i, w in enumerate(vocab_list)}

# 加载模型
model = FastInputModel(vocab_size=len(vocab_list)).to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))

# 执行预测
text = "深度学习"
result = predict(text, model, word2index, index2word, device)
print(f"预测结果: {result}")
# 输出: 预测结果: ['技术', '模型', '算法', '框架', '应用']
```

## 📈 查看训练日志

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 浏览器访问
# http://localhost:6006

```









