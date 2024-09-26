import gradio as gr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()

parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])

device = torch.device("mps")
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

history = pd.DataFrame(columns=['ppl', 'sentence']).astype({'sentence': 'str', 'ppl': 'float64'})
history = history._append({'sentence': "placeholder", 'ppl': 0}, ignore_index=True)

def calculate_perplexity(sentence, history):
    encodings = tokenizer(sentence, return_tensors="pt")
    print(encodings)
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss.item()
    
    history = history._append({'sentence': sentence, 'ppl': neg_log_likelihood}, ignore_index=True)

    return neg_log_likelihood, history

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Textbox(label="sentence", scale=3)
        submit_button = gr.Button("submit", scale=1)
    with gr.Row():
        output = gr.Textbox(label="ouput")
    hist = gr.DataFrame(type="pandas", value=history)
    
    submit_button.click(calculate_perplexity, [text, hist], [output, hist])
    
demo.launch()