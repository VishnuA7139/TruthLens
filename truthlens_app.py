import gradio as gr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import wikipedia
from sentence_transformers import SentenceTransformer, util

tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert_model")
model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_model")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

def get_wikipedia_summary(text):
    try:
        page = wikipedia.page(text, auto_suggest=False)
        return page.summary, page.url
    except Exception:
        try:
            search_results = wikipedia.search(text)
            if search_results:
                page = wikipedia.page(search_results[0])
                return page.summary, page.url
        except Exception:
            pass
    return "", ""

def predict(text):
    # Encode input
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0][pred_label].item()

    # Retrieve Wikipedia summary
    wiki_summary, wiki_url = get_wikipedia_summary(text)

    if wiki_summary:
        query_embedding = embedder.encode(text, convert_to_tensor=True)
        wiki_embedding = embedder.encode(wiki_summary, convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, wiki_embedding).item()
    else:
        similarity = 0.0

    # Decision logic with semantic fallback
    if similarity > 0.6:
        label = "Real"
        confidence = similarity
    else:
        label = "Fake" if pred_label == 1 else "Real"
        confidence = pred_prob

    return {
        "Prediction": f"{label} ({confidence:.2f})",
        "Wikipedia Summary": wiki_summary or "No summary found.",
        "Wikipedia Link": wiki_url or "N/A",
        "Semantic Similarity Score": f"{similarity:.2f}"
    }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news claim here..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Wikipedia Summary"),
        gr.Textbox(label="Wikipedia Link"),
        gr.Textbox(label="Semantic Similarity Score")
    ],
    title="TruthLens - Fake News Detection and Verification",
    description="This tool predicts whether a claim is likely Real or Fake and retrieves related Wikipedia content."
)

if __name__ == "__main__":
    interface.launch()
