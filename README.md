# 🧠 AI/ML Master Portfolio

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> End-to-end AI/ML portfolio covering Classical ML, Deep Learning, NLP, LLMs, RAG Systems and Production Deployment — built to demonstrate real-world skills for top-paying AI/ML roles.

---

## 🎯 Live Demo
👉 **[Click here to try the live demo](https://42cc49234fba769e17.gradio.live)**

---

## 🏆 Results Summary

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| 🚢 Titanic Survival | XGBoost + Optuna | CV Accuracy | 83.56% |
| 🖼️ CIFAR-10 Images | ResNet18 Transfer Learning | Accuracy | 40% (CPU) |
| 💬 IMDb Sentiment | DistilBERT fine-tuned | F1 Score | 85.60% |
| 🤖 RAG Q&A System | FLAN-T5 + FAISS | Status | Working ✅ |

---

## 📁 Project Structure
ai-ml-portfolio/
├── 📓 notebooks/
│   ├── 01_eda_classical_ml.ipynb     ← EDA + 6 ML Models + Optuna + SHAP
│   ├── 02_deep_learning_cnn.ipynb    ← ResNet18 CNN Transfer Learning
│   ├── 03_nlp_bert.ipynb             ← DistilBERT Fine-tuning on IMDb
│   └── 04_rag_system.ipynb           ← RAG System with FAISS + FLAN-T5
├── 🎮 gradio_demo/
│   └── app.py                        ← Live Gradio Demo App
├── 📦 models/saved/                  ← Saved trained models
├── 📊 data/processed/                ← EDA charts and results
├── 🐳 Dockerfile                     ← Container deployment
└── 📄 README.md

---

## 🛠️ Tech Stack

### Classical ML
- **Scikit-learn** — Pipelines, preprocessing, cross-validation
- **XGBoost, LightGBM, CatBoost** — Gradient boosting models
- **Optuna** — Bayesian hyperparameter optimization
- **SHAP** — Model explainability and feature importance

### Deep Learning
- **PyTorch** — Custom training loops, optimizers, schedulers
- **Torchvision** — ResNet18 transfer learning
- **CIFAR-10** — 10-class image classification

### NLP & LLMs
- **Hugging Face Transformers** — DistilBERT fine-tuning
- **Datasets** — IMDb sentiment dataset
- **FLAN-T5** — Text generation for RAG

### RAG System
- **LangChain** — Document loading and text splitting
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Text embeddings
- **PyPDF** — PDF ingestion

### MLOps & Deployment
- **Git & GitHub** — Version control and portfolio hosting
- **Docker** — Containerization
- **Gradio** — Live interactive demo
- **GitHub Actions** — CI/CD pipeline

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/VarshaSree5/ai-ml-portfolio.git
cd ai-ml-portfolio
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the live demo
```bash
python gradio_demo/app.py
```

### 5. Run notebooks
Open any notebook in `notebooks/` folder in VS Code with the venv kernel selected.

---

## 📓 Notebooks Guide

### 01 — EDA + Classical ML
- Exploratory Data Analysis on Titanic dataset
- Feature engineering (FamilySize, IsAlone, Title)
- 6 ML models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Hyperparameter tuning with Optuna (50 trials)
- SHAP feature importance visualization

### 02 — Deep Learning CNN
- CIFAR-10 dataset (50,000 training images, 10 classes)
- ResNet18 transfer learning with frozen backbone
- Custom classification head with dropout
- Training history visualization
- Sample predictions with correct/incorrect highlighting

### 03 — NLP with DistilBERT
- IMDb sentiment dataset (2000 train, 500 validation)
- DistilBERT tokenization and fine-tuning
- 85.20% accuracy, 85.60% F1 score
- Custom sentence inference pipeline

### 04 — RAG System
- PDF ingestion of "Attention is All You Need" paper
- Text chunking with RecursiveCharacterTextSplitter
- FAISS vector database with 93 vectors
- FLAN-T5 question answering
- End-to-end retrieval augmented generation

---

## 💼 Job Roles This Project Covers

| Role | Skills Demonstrated |
|------|-------------------|
| ML Engineer | Sklearn pipelines, XGBoost, model deployment |
| Data Scientist | EDA, feature engineering, model evaluation |
| NLP Engineer | BERT fine-tuning, tokenization, text classification |
| LLM Engineer | RAG systems, FAISS, FLAN-T5, LangChain |
| Computer Vision Engineer | CNN, transfer learning, image classification |
| MLOps Engineer | Docker, GitHub Actions, model serving |

---

## 📄 License
MIT License — feel free to use this project as a reference.

---

## 👩‍💻 Author
**Varsha Sree**  
GitHub: [@VarshaSree5](https://github.com/VarshaSree5)