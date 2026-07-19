# AI Learning

暂为个人的深度学习项目

## 章节

| # | 笔记本 | 内容 |
|---|--------|------|
| 1 | `ch01_python.ipynb` | Python 基础与环境配置 |
| 2 | `ch02_ml.ipynb` | 机器学习基础概念 |
| 3 | `ch03_dl.ipynb` | 深度学习基础 |
| 4 | `ch04_evaluation.ipynb` | 模型评估与正则化 |
| 5 | `ch05_pytorch.ipynb` | PyTorch 入门 |
| 6 | `ch06_cnn.ipynb` | 卷积神经网络 |
| 7 | `ch07_rnn.ipynb` | 循环神经网络 |
| 8 | `ch08_llm.ipynb` | 大语言模型与对齐技术 |
| 9 | `ch09_pca.ipynb` | PCA 主成分分析 |
| 10 | `ch10_ddcgan.ipynb` | 非线性降维与生成模型 |

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
