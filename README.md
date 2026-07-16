# AI Learning — 人工智能系统学习项目

从 Python 语法到 LLM Agent 部署，14 章系统课程覆盖 AI 学习全链路：基础编程 → 经典机器学习/深度学习 → 现代 AI 工程（RAG/Agent/多模态/MLOps）。

## 📚 章节

| # | 笔记本 | 内容 | 大小 |
|---|--------|------|------|
| 1 | `ch01_python_basics.ipynb` | Python 基础语法（变量/数据结构/控制流/函数/OOP + C++ 对比） | 22K |
| 2 | `ch02_conda_packages.ipynb` | Conda 与 Python 包管理（venv/conda/pip/依赖管理最佳实践） | 17K |
| 3 | `ch03_deep_learning_basics.ipynb` | 深度学习基础（反向传播/GD变体/权重初始化/BatchNorm/MLP/MNIST/统计ML） | 39K |
| 4 | `ch04_model_evaluation.ipynb` | 模型评估与正则化（bias-variance/交叉验证/L1-L2/Dropout/ROC-AUC/学习曲线） | 22K |
| 5 | `ch05_pytorch_intro.ipynb` | PyTorch 入门（Tensor/autograd/nn.Module/Dataset/训练循环/模型保存） | 21K |
| 6 | `ch06_cifar10_cnn.ipynb` | CIFAR-10 分类实战（CNN 架构/数据增强/迁移学习/混淆矩阵分析） | 24K |
| 7 | `ch07_sequence_nlp.ipynb` | 序列模型与 NLP 基础（RNN/LSTM/GRU 从零实现/tokenization/词嵌入/文本生成） | 20K |
| 8 | `ch08_llm_sft_rl.ipynb` | LLM 与对齐技术（Transformer/GPT/SFT+LoRA实战/RLHF/DPO/Prompt Engineering） | 21K |
| 9 | `ch09_pca.ipynb` | PCA 主成分分析（数学推导/特征值分解/Digits 降维重建/方差解释） | 16K |
| 10 | `ch10_tsne_vae_ddcgan.ipynb` | 非线性降维与生成模型（t-SNE/UMAP/Autoencoder/VAE/DDcGAN） | 47K |
| 11 | `ch11_rag.ipynb` | RAG 系统实战（Embedding/向量数据库/检索管道/Self-RAG/Graph RAG/RAGAS评估） | 9K |
| 12 | `ch12_agent.ipynb` | AI Agent 系统（ReAct/Tool Use/多Agent架构/LangGraph/Agent评估） | 7K |
| 13 | `ch13_multimodal.ipynb` | 多模态 AI（CLIP/GPT-4V/Whisper/TTS/多模态RAG） | 4K |
| 14 | `ch14_mlops.ipynb` | MLOps 与模型部署（ONNX/量化/FastAPI/Docker/vLLM/生产监控） | 5K |

## 🎯 学习路线

```
L1 基础    Ch1(Python) → Ch2(环境)
              ↓
L2 经典    Ch3(DL基础) → Ch4(模型评估) → Ch5(PyTorch) → Ch6(CIFAR-10)
              ↓                                        ↓
           Ch7(RNN/NLP)                              Ch9(PCA)
              ↓                                        ↓
L3 现代    Ch8(LLM+对齐+Prompt) ←─────────────── Ch10(t-SNE/VAE/GAN)
              ↓
           Ch11(RAG) → Ch12(Agent) → Ch13(多模态)
              ↓
工程       Ch14(MLOps/部署)
```

- **无基础**：从 Ch1 逐章执行
- **有 Python 基础**：从 Ch3 开始
- **有 DL 基础**：Ch8→Ch11→Ch12→Ch13 现代 AI 主线
- **关注工程落地**：Ch14 独立成章

## 📖 各章概要

### 第 1 章：Python 基础语法
变量/类型、四种容器、推导式与切片、控制流、函数与 lambda、OOP（手写 DenseLayer）、综合练习：DataLoader。每节附带 C++ 等价代码对比。

### 第 2 章：Conda 与包管理
venv/pip/conda 三件套、镜像源、requirements.txt vs environment.yml、常见问题排查

### 第 3 章：深度学习基础
感知机、激活函数（Sigmoid/ReLU/Tanh/Softmax + 可视化）、梯度消失理论、损失函数（MSE/BCE/CE）、梯度下降可视化、SGD/Momentum/Adam 对比（Rosenbrock 函数优化）、反向传播 8 步手动推导 + 计算图视角、线性/逻辑回归、MLP（任意深度）、权重初始化（Xavier vs He + 激活值标准差验证）、BatchNorm 原理、MNIST 纯 NumPy 实现、统计 ML（SVM/树/森林/KNN）对比

### 第 4 章：模型评估与正则化
bias-variance 权衡（多项式回归可视化）、K-Fold 交叉验证（Ridge alpha 选择）、L1/L2/Dropout/ElasticNet 对比、混淆矩阵 + Precision/Recall/F1/ROC-AUC（不平衡数据的警示）、学习曲线（欠拟合/过拟合诊断）、Early Stopping

### 第 5 章：PyTorch 入门
Tensor + GPU、Autograd 自动求导（梯度下降可视化）、nn.Module（Sequential + 自定义模块）、Dataset/DataLoader、标准训练循环模板、模型保存/加载

### 第 6 章：CIFAR-10 实战
卷积的数学本质（稀疏连接/参数共享/特征层级）、池化层理论（平移不变性/感受野）、数据增强（RandomCrop/Flip/ColorJitter）、VGG 风格 3-block CNN + BatchNorm + Dropout、Kaiming 初始化、OneCycleLR 学习率调度、混淆矩阵 + 分类报告、迁移学习 ResNet-18

### 第 7 章：序列模型与 NLP 基础
RNN Cell 手写实现、BPTT 梯度消失推导、LSTM（三门 + 细胞状态）从零实现与 PyTorch 版、GRU 对比、Tokenization（word/char/BPE）、Word2Vec/GloVe 词嵌入、LSTM 字符级语言模型（完整训练 + 文本生成）、序列模型全景对比、经典 NLP 任务体系

### 第 8 章：LLM 与对齐
Transformer 架构详解、Self-Attention 信息检索视角（Q/K/V + √dₖ 缩放原理）、因果掩码/位置编码可视化、GPT 系列发展、SFT（ChatML + Loss Masking + TRL SFTTrainer+LoRA 实战）、RLHF 三阶段（奖励模型 + PPO）、DPO 损失函数实现、HuggingFace 推理、Prompt Engineering（Zero-shot/Few-shot/CoT/结构化输出/System Prompt/注入防御）

### 第 9 章：PCA 主成分分析
PCA 三种等价视角（最大方差/最小重建误差/最大分离度）、中心化 → 协方差矩阵 → 特征值分解完整推导、Digits 降维重建（1/2/4/8/16/32/48/64 PCs 对比）、累积方差解释曲线

### 第 10 章：非线性降维与生成模型

**t-SNE**：完整推导（高斯核→t 分布→KL 散度）、UMAP 对比、流形假设与潜在空间理论。**自编码器**：MLP-AE 递进，潜在空间的连续性与流形假设。**VAE**：reparameterization trick + 生成采样，ELBO 分解为重建损失 + KL 正则。

**DDcGAN**：GAN 基础（Minimax 博弈/模式坍塌）、双判别器架构设计（全局判别器 + 局部判别器）、Hinge Loss 公式推导、Projection Discriminator 条件注入。完整 PyTorch 实现：Generator + GlobalDiscriminator + LocalDiscriminator + DDcGANLoss。MNIST 训练演示 + 生成结果可视化。

### 第 11 章：RAG 系统实战
Embedding 语义搜索（对比学习/InfoNCE）、向量数据库（Chroma/HNSW/IVF/混合检索）、RAG 三阶段（Chunking 策略/检索→重排序→生成）、高级 RAG（Self-RAG/CRAG/Graph RAG）、RAGAS 评估（Faithfulness/Answer Relevance/Context Recall）

### 第 12 章：AI Agent 系统
ReAct 模式（Thought→Action→Observation 循环）、记忆系统（短期/工作/长期）、Plan-and-Execute、Tool Use/Function Calling 协议、工具设计原则、多 Agent 架构（层级/流水线/辩论式）、LangGraph 有向图编排、Agent 评估指标

### 第 13 章：多模态 AI
CLIP 对比学习（共享嵌入空间/零样本分类）、GPT-4V/Gemini Vision 架构、Whisper STT（模型选择/语言支持）、TTS 语音合成、多模态 RAG（跨模态检索 + 多模态 LLM）、ImageBind 跨模态 embedding

### 第 14 章：MLOps 与模型部署
模型导出（ONNX/TorchScript/GGUF）、量化（INT8/INT4/GPTQ/AWQ）、FastAPI 模型服务化、Docker 容器化 + GPU 支持、LLM 推理引擎（vLLM/Ollama/llama.cpp/TGI）、生产监控（延迟/吞吐量/数据漂移）、反馈循环与持续迭代

## 🚀 快速开始

```bash
cd AI-learning
python3 -m venv .venv
source .venv/bin/activate
uv pip install jupyter numpy matplotlib scikit-learn seaborn tqdm umap-learn pandas notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
jupyter notebook notebooks/
```

## 🔧 环境要求

- Python 3.10+
- PyTorch 2.0+, torchvision
- NumPy, Matplotlib, scikit-learn, seaborn, tqdm
- umap-learn (Ch10)
- transformers (Ch8, 可选)
- sentence-transformers, chromadb (Ch11, 可选)
