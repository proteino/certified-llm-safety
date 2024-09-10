import gradio as gr
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from argparse import ArgumentParser
import kagglehub
import torch

parser = ArgumentParser()

parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])

args = parser.parse_args()
device = torch.device(args.device)


# load model 
path = kagglehub.model_download("grossjct/prompt_classifier/pyTorch/default", "distilbert.pt")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

labels = {
    "LABEL_0": "harmful",
    "LABEL_1": "safe"
}


def predict(prompt):
    out = pipe(prompt)[0]
    return labels[out["label"]], out["score"]


with gr.Blocks() as demo:
    with gr.Row():
        prompt = gr.Textbox(label="prompt", scale=3)
        submit_button = gr.Button("submit", scale=1)
    with gr.Row():
        label = gr.Textbox(label="label")
        score = gr.Textbox(label="score")
    
    submit_button.click(predict, [prompt], [label, score])

demo.launch()