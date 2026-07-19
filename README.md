# AI Learning

暂为个人的深度学习项目

## 章节

| # | 笔记本 | 内容 | 大小 |
|---|--------|------|------|
| 1 | `ch01_python_basics.ipynb` | Python 基础与环境配置（语法 + C++ 对比 + conda/pip/venv） | 35K |
| 2 | `ch02_ml_fundamentals.ipynb` | 机器学习基础（27 个核心概念的统一定义：数学前提 + ML 术语 + 补充） | 11K |
| 3 | `ch03_deep_learning_basics.ipynb` | 深度学习基础（感知机/激活函数/损失函数/梯度下降/反向传播/MLP/MNIST 纯 NumPy） | 42K |
| 4 | `ch04_model_evaluation.ipynb` | 模型评估与正则化（偏差-方差分解/交叉验证/L1-L2/Dropout/ROC-AUC/学习曲线/贝叶斯视角） | 22K |
| 5 | `ch05_pytorch_intro.ipynb` | PyTorch 入门（Tensor/Autograd 计算图/nn.Module/Dataset 协议/训练循环/模型保存） | 21K |
| 6 | `ch06_cifar10_cnn.ipynb` | 卷积神经网络：从图像理解空间（CNN 架构/卷积数学本质/池化/感受野/BN/迁移学习） | 22K |
| 7 | `ch07_sequence_nlp.ipynb` | 循环神经网络：从序列理解时间（RNN/LSTM/BPTT/GRU/tokenization/词嵌入/NLP 任务全景） | 17K |
| 8 | `ch08_llm_sft_rl.ipynb` | LLM 与对齐技术（Transformer/Self-Attention/GPT/SFT+LoRA/RLHF/DPO/Prompt Engineering） | 19K |
| 9 | `ch09_pca.ipynb` | PCA 主成分分析（协方差矩阵/特征值分解/拉格朗日推导/降维与重建/方差解释率） | 14K |
| 10 | `ch10_tsne_vae_ddcgan.ipynb` | 非线性降维与生成模型（t-SNE/UMAP/Autoencoder/VAE/GAN/DDcGAN 双判别器） | 34K |
| 11 | `ch11_rag.ipynb` | RAG 系统实战（Embedding/向量数据库/检索管道/Self-RAG/Graph RAG/RAGAS 评估） | 6K |
| 12 | `ch12_agent.ipynb` | AI Agent 系统（ReAct/Tool Use/多 Agent 架构/LangGraph/Agent 评估） | 4K |
| 13 | `ch13_multimodal.ipynb` | 多模态 AI（CLIP/GPT-4V/Whisper/TTS/多模态 RAG） | 2K |
| 14 | `ch14_mlops.ipynb` | MLOps 与模型部署（ONNX/量化/FastAPI/Docker/vLLM/生产监控） | 4K |

## 学习路线

```
L1 入门    Ch1(Python+环境) → Ch2(ML 基础)
                ↓
L2 核心    Ch3(DL 基础) → Ch4(模型评估) → Ch5(PyTorch)
                ↓
           Ch6(CNN: 空间) → Ch7(RNN: 时间)
                ↓               ↓
           Ch9(PCA)          Ch8(LLM+对齐)
                ↓               ↓
           Ch10(降维与生成) ←───┘
                ↓
L3 工程    Ch11(RAG) → Ch12(Agent) → Ch13(多模态)
                ↓
           Ch14(MLOps/部署)
```

## 各章概要

### 第 1 章：Python 基础与环境配置
变量/类型/容器/控制流/函数/OOP + 每节 C++ 对比。虚拟环境（venv/conda/pip）、镜像源、requirements.txt/environment.yml、依赖管理最佳实践。

### 第 2 章：机器学习基础
27 个核心概念的统一入口。三组：(1) 数学前提——向量/矩阵、梯度/链式法则、概率/期望/方差、MLE、熵/KL/交叉熵、拉格朗日乘子、特征值/特征向量、L1/L2 范数；(2) ML 术语——监督/无监督、样本/特征/标签/维度、train/val/test、epoch/batch/iteration、参数/超参数、损失/优化器、过拟合/欠拟合、归一化/标准化、评估指标、Softmax/Sigmoid；(3) 补充——前向/反向、生成/判别模型、Attention、Embedding、聚类、感受野、t 分布。

### 第 3 章：深度学习基础
神经网络定义（三种基本形态：MLP/CNN/RNN）。感知机、激活函数（为什么必须非线性 + 线性退化证明）、损失函数（MSE/BCE/CE + CE-Softmax 梯度推导）、梯度下降（1D→2D→高维直觉）、反向传播（链式法则 + 2-3-1 网络数值手算示例）、线性/逻辑回归、MLP（全连接层的矩阵数学）、MNIST 纯 NumPy 实现、统计 ML 方法概览。

### 第 4 章：模型评估与正则化
偏差-方差分解代数推导（期望展开到 σ²+Bias²+Var）、交叉验证、L1/L2/Dropout 正则化（L1 稀疏性几何视角）、分类评估指标（混淆矩阵/Precision/Recall/F1/ROC-AUC + 准确率陷阱数字案例）、学习曲线、Early Stopping、贝叶斯视角（不确定性量化统一语言）。

### 第 5 章：PyTorch 入门
Tensor 数据结构（标量/向量/矩阵/张量的数学对应）、Autograd 动态计算图机制、梯度清零原理（zero_grad）、nn.Module 设计哲学（`__init__` vs `forward` 分离）、Dataset/DataLoader 底层协议（`__getitem__`/`__len__`/collate_fn）、标准训练循环（每行代码的作用）、模型保存（state_dict vs 完整模型）。

### 第 6 章：卷积神经网络——从图像理解空间
CNN 架构设计（为什么 MLP 不适合图像：空间结构破坏 + 参数爆炸，896 vs 157 万）、卷积数学本质（互相关/稀疏连接/参数共享/特征层级）、池化（平移容忍/增大感受野）、批归一化、感受野递推公式与数值例子、训练策略、迁移学习（CNN 局限→引出 Ch7 RNN）。

### 第 7 章：循环神经网络——从序列理解时间
从 CNN 的局限出发（空间不变量→时间不变量→参数共享的推广）。RNN 核心机制（隐藏状态/权重共享/BPTT 3 步链式展开）、LSTM（三门+细胞状态/遗忘门消除梯度消失推导/GRU 工程对比）、NLP 基础（tokenization/词嵌入/Skip-gram 目标函数/序列模型全景对比/NLP 任务体系）。

### 第 8 章：LLM 与对齐技术
Transformer 架构详解（Self-Attention Q/K/V 角色 + O(n²d) 复杂度分析 + 信息检索视角）、GPT 系列发展、SFT 监督微调（ChatML/Loss Masking/TRL SFTTrainer+LoRA 实战）、RLHF 三阶段（奖励模型/PPO）、DPO 损失函数推导（Bradley-Terry 偏好模型）、Prompt Engineering（zero-shot/few-shot/CoT/结构化输出/System Prompt/注入防御）。

### 第 9 章：PCA 主成分分析
PCA 几何直觉（方差最大的方向=特征向量，特征值=方差大小）。完整数学推导：数据中心化→协方差矩阵→拉格朗日乘子法构造与求解→特征方程→降维→重建（k=1 特例到 k 维推广）→方差解释率。Digits 降维重建与累积方差解释曲线。

### 第 10 章：非线性降维与生成模型
t-SNE（高斯核→t 分布→KL 散度最小化/困惑度/crowding problem）、UMAP 对比、Autoencoder（潜在空间与流形假设）、VAE（reparameterization trick/ELBO）。GAN 基础（生成器与判别器的架构级定义/极小极大博弈/最优判别器推导/JS 散度等价/梯度消失数学根源）。DDcGAN 重点：单判别器的感受野冲突→双判别器分而治之（全局+局部）/Hinge Loss 梯度机制/投影判别器内积结构/随机裁剪策略/训练平衡三手段。完整 PyTorch 实现与 MNIST 训练演示。

### 第 11 章：RAG 系统实战
Embedding 语义搜索、向量数据库（Chroma/HNSW/IVF）、RAG 管道（Chunking/检索/重排序/生成）、高级 RAG（Self-RAG/CRAG/Graph RAG）、RAGAS 评估。

### 第 12 章：AI Agent 系统
ReAct 模式、记忆系统、Tool Use/Function Calling、多 Agent 架构（层级/流水线/辩论式）、LangGraph 编排、Agent 评估。

### 第 13 章：多模态 AI
CLIP 对比学习、GPT-4V/Gemini Vision、Whisper STT、TTS、多模态 RAG。

### 第 14 章：MLOps 与模型部署
模型导出（ONNX/量化）、FastAPI 服务化、Docker 容器化 + GPU 支持、LLM 推理引擎（vLLM/Ollama）、生产监控。

## 快速开始

```bash
cd AI-learning
python3 -m venv .venv
source .venv/bin/activate
uv pip install jupyter numpy matplotlib scikit-learn seaborn tqdm umap-learn pandas notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
jupyter notebook notebooks/
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+, torchvision
- NumPy, Matplotlib, scikit-learn, seaborn, tqdm
- umap-learn (Ch10)
- transformers (Ch8, 可选)
- sentence-transformers, chromadb (Ch11, 可选)
