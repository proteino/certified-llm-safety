import gradio as gr
from datasets import load_dataset
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification
from argparse import ArgumentParser
import kagglehub
import torch

parser = ArgumentParser()

parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])
parser.add_argument("--categories", "-c", action="store_true")
parser.add_argument("--distil", action="store_true")
parser.add_argument("--model_path", "-m", type=str, required=True)

args = parser.parse_args()
device = torch.device(args.device)
path = args.model_path

if args.categories:
    harmful_prompts_dataset = load_dataset("Babelscape/ALERT", "alert", split="test") # ignore the split, dataset only has test split
    data_frame = harmful_prompts_dataset.to_pandas()
    categories = data_frame.category.unique()
    n_labels = len(categories) + 1
    label_to_category = {f"LABEL_{i+1}": text for i, text in enumerate(categories)}
    label_to_category["LABEL_0"] = "safe"
    
    
else:
    n_labels = 2
    label_to_category = {
        "LABEL_0": "harmful",
        "LABEL_1": "safe"
    }


if args.distil:
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=n_labels)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
else:
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # pass the pre-trained DistilBert to our define architecture
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=n_labels)

model.load_state_dict(torch.load(path, map_location=device))
model.eval()
pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)



def predict(prompt):
    out = pipe(prompt)[0]
    return label_to_category[out["label"]], out["score"]


with gr.Blocks() as demo:
    with gr.Row():
        prompt = gr.Textbox(label="prompt", scale=3)
        submit_button = gr.Button("submit", scale=1)
    with gr.Row():
        label = gr.Textbox(label="label")
        score = gr.Textbox(label="score")
    
    submit_button.click(predict, [prompt], [label, score])

demo.launch()