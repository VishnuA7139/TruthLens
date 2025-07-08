import gradio as gr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import wikipedia
from sentence_transformers import SentenceTransformer, util

# Load model and tokenizer
model_path = "./models/distilbert_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def classify_and_explain(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()
        label = "Real" if prediction == 0 else "Fake"

    # Wikipedia retrieval
    try:
        summary = wikipedia.summary(text, sentences=2)
        wiki_page = wikipedia.page(text)
        wiki_link = wiki_page.url
    except Exception:
        summary = "No related Wikipedia content found."
        wiki_link = ""

    # Semantic similarity
    try:
        claim_emb = embedder.encode(text, convert_to_tensor=True)
        summary_emb = embedder.encode(summary, convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(claim_emb, summary_emb).item()
        sim_text = f"Semantic Similarity: {sim_score:.2f}"
    except Exception:
        sim_text = "Semantic similarity unavailable."

    label_color = "green" if label == "Real" else "red"
    output_label = f"<span style='color:{label_color}; font-weight:bold'>{label}</span>"

    wiki_html = (
        f"<p>{summary}</p><p><a href='{wiki_link}' target='_blank'>Read more</a></p>"
        if wiki_link else f"<p>{summary}</p>"
    )

    return output_label, wiki_html, sim_text

iface = gr.Interface(
    fn=classify_and_explain,
    inputs=gr.Textbox(label="Enter News Statement"),
    outputs=[
        gr.HTML(label="Prediction"),
        gr.HTML(label="Wikipedia Summary"),
        gr.Textbox(label="Semantic Similarity")
    ],
    title="TruthLens - Fake News Detector",
    description="Enter a claim to classify it and get supporting Wikipedia content.",
)

iface.launch()
