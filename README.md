# AI Learning — 人工智能系统学习项目

从 Python 语法到 LLM 对齐，对标 Stanford CS229/CS231n、Microsoft AI-For-Beginners、Chollet《Deep Learning with Python》等主流课程体系，覆盖 AI 学习全链路。

## 📚 章节

| # | 笔记本 | 内容 | 大小 |
|---|--------|------|------|
| 1 | `ch01_python_basics.ipynb` | Python 基础语法（变量/数据结构/控制流/函数/OOP/文件操作） | 22K |
| 2 | `ch02_conda_packages.ipynb` | Conda 与 Python 包管理（venv/conda/pip/依赖管理最佳实践） | 17K |
| 3 | `ch03_model_evaluation.ipynb` | 模型评估与正则化（bias-variance/交叉验证/L1-L2/Dropout/ROC-AUC/学习曲线） | 22K |
| 4 | `ch04_deep_learning_basics.ipynb` | 深度学习基础（反向传播/回归分类/MLP/MNIST 手写识别/统计 ML） | 37K |
| 5 | `ch05_pytorch_intro.ipynb` | PyTorch 入门（Tensor/autograd/nn.Module/训练循环/模型保存） | 21K |
| 6 | `ch06_cifar10_cnn.ipynb` | CIFAR-10 分类实战（CNN 架构/数据增强/迁移学习/混淆矩阵分析） | 24K |
| 7 | `ch07_sequence_nlp.ipynb` | 序列模型与 NLP 基础（RNN/LSTM/GRU 从零实现/tokenization/词嵌入/文本生成） | 20K |
| 8 | `ch08_llm_sft_rl.ipynb` | LLM 与对齐技术（Transformer/GPT/SFT/RLHF/DPO/HuggingFace） | 21K |
| 9 | `ch09_tsne_vae.ipynb` | 非线性降维与生成模型（t-SNE/UMAP/Autoencoder/VAE） | 18K |
| 10 | `ch10_pca_ddcgan.ipynb` | **PCA 与 DDcGAN 重点专题**（数学推导/手写实现/双判别器架构/完整训练） | 40K |

## 🚀 快速开始

```bash
cd AI-learning
python3 -m venv .venv
source .venv/bin/activate
uv pip install jupyter numpy matplotlib scikit-learn seaborn tqdm umap-learn pandas notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
jupyter notebook notebooks/
```

## 📖 各章概要

### 第 1 章：Python 基础语法
变量/类型、四种容器、推导式与切片、控制流、函数与 lambda、OOP（手写 DenseLayer）、综合练习：DataLoader

### 第 2 章：Conda 与包管理
venv/pip/conda 三件套、镜像源、requirements.txt vs environment.yml、常见问题排查

### 第 3 章：模型评估与正则化
bias-variance 权衡（多项式回归可视化）、K-Fold 交叉验证（Ridge alpha 选择）、L1/L2/Dropout/ElasticNet 对比、混淆矩阵 + Precision/Recall/F1/ROC-AUC（不平衡数据的警示）、学习曲线（欠拟合/过拟合诊断）、Early Stopping

### 第 4 章：深度学习基础
感知机、激活函数（Sigmoid/ReLU/Tanh/Softmax + 可视化）、损失函数（MSE/BCE/CE）、梯度下降可视化、反向传播 8 步手动推导、线性/逻辑回归、MLP（任意深度）、MNIST 纯 NumPy 实现、统计 ML（SVM/树/森林/KNN）对比

### 第 5 章：PyTorch 入门
Tensor + GPU、Autograd 自动求导（梯度下降可视化）、nn.Module（Sequential + 自定义模块）、Dataset/DataLoader、标准训练循环模板、模型保存/加载

### 第 6 章：CIFAR-10 实战
数据增强（RandomCrop/Flip/ColorJitter）、VGG 风格 3-block CNN + BatchNorm + Dropout、Kaiming 初始化、OneCycleLR 学习率调度、混淆矩阵 + 分类报告、迁移学习 ResNet-18

### 第 7 章：序列模型与 NLP 基础
RNN Cell 手写实现、BPTT 梯度消失推导、LSTM（三门 + 细胞状态）从零实现与 PyTorch 版、GRU 对比、Tokenization（word/char/BPE）、Word2Vec/GloVe 词嵌入、LSTM 字符级语言模型（完整训练 + 文本生成）、序列模型全景对比、经典 NLP 任务体系

### 第 8 章：LLM 与对齐
Transformer 架构详解、Self-Attention 公式推导 + 手写实现、因果掩码/位置编码可视化、GPT 系列发展、SFT（ChatML + Loss Masking）、RLHF 三阶段（奖励模型 + PPO）、DPO 损失函数实现、HuggingFace 推理

### 第 9 章：非线性降维与生成模型
t-SNE（原理 + 与 PCA 可视化对比 + perplexity 影响）、UMAP（速度/可复现性优势）、Autoencoder（MLP-AE → Conv-AE 递进）、VAE（reparameterization trick + 生成采样）

### 第 10 章（重点）：PCA & DDcGAN

**PCA**：中心化 → 协方差矩阵 → 特征值分解完整推导、Digits 重建（1/2/4/8/16/32/48/64 PCs 对比）、累积方差解释曲线。

**DDcGAN**：GAN 基础（Minimax 博弈/模式坍塌）、双判别器架构设计（全局判别器 + 局部判别器）、Hinge Loss 公式推导、Projection Discriminator 条件注入。完整 PyTorch 实现：Generator + GlobalDiscriminator + LocalDiscriminator + DDcGANLoss。MNIST 训练演示 + 生成结果可视化。

## 🎯 学习路线

```
Ch1(Python) → Ch2(环境) → Ch3(评估/正则化)
    ↓
Ch4(DL基础) → Ch5(PyTorch) → Ch6(CNN/CIFAR-10)
    ↓
Ch7(RNN/NLP) → Ch8(LLM/SFT/RLHF)
    ↓
Ch9(t-SNE/VAE) → Ch10(PCA/DDcGAN) ★ 重点
```

- **无基础**：从 Ch1 逐章执行
- **有 Python 基础**：从 Ch3 开始
- **有 DL 基础**：Ch6→Ch7→Ch8 主线，Ch9→Ch10 进阶

## 📊 项目统计

- **10 个 notebook**，~260KB 内容
- **175 个 cell**，全部可运行
- **70+ 数学公式**（LaTeX 渲染）
- **40+ 张可视化图片**

## 🔧 环境要求

- Python 3.10+
- PyTorch 2.0+, torchvision
- NumPy, Matplotlib, scikit-learn, seaborn, tqdm
- umap-learn (Ch9)
- transformers (Ch8, 可选)
