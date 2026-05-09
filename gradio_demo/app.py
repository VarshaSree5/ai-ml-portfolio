import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.eval()

print("Model loaded successfully!")


def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    pos = probs[0][1].item()
    neg = probs[0][0].item()
    return pos, neg


def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text!"
    pos, neg = get_sentiment(text)
    label = "POSITIVE" if pos > neg else "NEGATIVE"
    emoji = "😊" if pos > neg else "😞"
    confidence = max(pos, neg)
    return (
        f"{emoji} {label}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Positive Score: {pos:.1%}\n"
        f"Negative Score: {neg:.1%}"
    )


def classify_review(text):
    if not text.strip():
        return "Please enter a review!"
    pos, neg = get_sentiment(text)
    label = "POSITIVE" if pos > neg else "NEGATIVE"
    emoji = "😊" if pos > neg else "😞"
    confidence = max(pos, neg)

    if pos > 0.8:
        verdict = "Strongly Positive"
    elif pos > 0.6:
        verdict = "Mildly Positive"
    elif neg > 0.8:
        verdict = "Strongly Negative"
    elif neg > 0.6:
        verdict = "Mildly Negative"
    else:
        verdict = "Neutral / Mixed"

    return (
        f"Prediction: {emoji} {label}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Positive: {pos:.1%}\n"
        f"Negative: {neg:.1%}\n\n"
        f"Verdict: {verdict}"
    )


def batch_analyze(text):
    if not text.strip():
        return "Please enter at least one review!"
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    lines = lines[:5]
    output = f"Analyzing {len(lines)} review(s)...\n"
    output += "=" * 40 + "\n\n"
    for i, line in enumerate(lines, 1):
        pos, neg = get_sentiment(line)
        label = "POSITIVE" if pos > neg else "NEGATIVE"
        emoji = "😊" if pos > neg else "😞"
        confidence = max(pos, neg)
        short = line[:60] + "..." if len(line) > 60 else line
        output += f"Review {i}: {emoji} {label} ({confidence:.1%})\n"
        output += f"Text: {short}\n\n"
    return output


def show_project_info():
    return """
PROJECT: AI/ML Master Portfolio
================================

MODELS BUILT:
1. Classical ML  - XGBoost + Optuna       - 83.56% accuracy
2. Deep Learning - ResNet18 CNN            - 40% (CPU only)
3. NLP           - DistilBERT fine-tuned   - 85.20% accuracy
4. RAG System    - FLAN-T5 + FAISS         - Working

TECH STACK:
- Classical ML   : Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
- Deep Learning  : PyTorch, ResNet18 Transfer Learning
- NLP / LLMs     : Hugging Face Transformers, DistilBERT, FLAN-T5
- RAG            : LangChain, FAISS, Sentence Transformers
- Deployment     : Gradio, Docker, GitHub Actions

GITHUB:
https://github.com/VarshaSree5/ai-ml-portfolio
"""


with gr.Blocks(theme=gr.themes.Soft(), title="AI/ML Portfolio Demo") as demo:

    gr.Markdown("""
    # 🧠 AI/ML Master Portfolio — Live Demo
    **Built with:** PyTorch · Transformers · Scikit-learn · FAISS · LangChain
    **GitHub:** https://github.com/VarshaSree5/ai-ml-portfolio
    ---
    """)

    with gr.Tab("😊 Sentiment Analysis"):
        gr.Markdown("### Analyze the sentiment of any text using DistilBERT")
        with gr.Row():
            with gr.Column(scale=1):
                sent_input = gr.Textbox(
                    lines=5,
                    label="Input Text",
                    placeholder="Type any sentence here..."
                )
                sent_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            with gr.Column(scale=1):
                sent_output = gr.Textbox(
                    lines=6,
                    label="Result"
                )
        gr.Examples(
            examples=[
                "This movie was absolutely fantastic!",
                "I hated every minute of it.",
                "The food was okay but nothing special.",
                "Best experience of my life!",
                "Terrible service, never coming back."
            ],
            inputs=sent_input
        )
        sent_btn.click(fn=analyze_sentiment, inputs=sent_input, outputs=sent_output)

    with gr.Tab("🎭 Review Classifier"):
        gr.Markdown("### Get detailed classification breakdown for any review")
        with gr.Row():
            with gr.Column(scale=1):
                rev_input = gr.Textbox(
                    lines=5,
                    label="Review Text",
                    placeholder="Paste a product or movie review here..."
                )
                rev_btn = gr.Button("🔍 Classify", variant="primary", size="lg")
            with gr.Column(scale=1):
                rev_output = gr.Textbox(
                    lines=8,
                    label="Detailed Result"
                )
        gr.Examples(
            examples=[
                "Outstanding product! Exceeded all my expectations.",
                "Complete waste of money. Broke after one day.",
                "It is fine, does what it says but nothing special."
            ],
            inputs=rev_input
        )
        rev_btn.click(fn=classify_review, inputs=rev_input, outputs=rev_output)

    with gr.Tab("📋 Batch Analysis"):
        gr.Markdown("### Analyze up to 5 reviews at once — one per line")
        with gr.Row():
            with gr.Column(scale=1):
                batch_input = gr.Textbox(
                    lines=7,
                    label="Multiple Reviews (one per line)",
                    placeholder="This is great!\nTerrible product.\nAverage quality.\nLoved it!\nWould not recommend."
                )
                batch_btn = gr.Button("🔍 Analyze All", variant="primary", size="lg")
            with gr.Column(scale=1):
                batch_output = gr.Textbox(
                    lines=12,
                    label="All Results"
                )
        batch_btn.click(fn=batch_analyze, inputs=batch_input, outputs=batch_output)

    with gr.Tab("📊 Project Results"):
        gr.Markdown("### Full project summary and results")
        info_btn = gr.Button("📄 Show Project Info", variant="secondary")
        info_output = gr.Textbox(lines=30, label="Project Information")
        info_btn.click(fn=show_project_info, inputs=[], outputs=info_output)

        gr.Markdown("""
        ## 🏆 Performance Table

        | Task | Model | Metric | Score |
        |------|-------|--------|-------|
        | 🚢 Titanic Survival | XGBoost + Optuna | CV Accuracy | 83.56% |
        | 🖼️ CIFAR-10 | ResNet18 Transfer | Accuracy | 40% (CPU) |
        | 💬 IMDb Sentiment | DistilBERT | F1 Score | 85.60% |
        | 🤖 RAG Q&A | FLAN-T5 + FAISS | Status | Working ✅ |

        ## 📁 Notebooks
        1. `01_eda_classical_ml.ipynb` — EDA + 6 ML Models + SHAP
        2. `02_deep_learning_cnn.ipynb` — ResNet18 CNN on CIFAR-10
        3. `03_nlp_bert.ipynb` — DistilBERT Fine-tuning on IMDb
        4. `04_rag_system.ipynb` — RAG with FAISS + FLAN-T5
        """)

demo.launch(share=True)