# evaluate model trained on AdvBench on the alert dataset 

import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset
import torch
import argparse
import kagglehub
from tqdm.auto import tqdm
import json

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
# parser.add_argument('--eval_type', type=str, choices=["harmful", "safe"])
parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])


args = parser.parse_args()

device = torch.device(args.device)

# Using custom classifier for safety filter
# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Load model weights
# path = kagglehub.model_download("grossjct/safety-filter-insertion/pyTorch/default", "distilbert_insertion-2.pt")
path = kagglehub.model_download("grossjct/prompt_classifier/pyTorch/default", "distilbert.pt")

model.load_state_dict(torch.load(path, map_location=device))
model.eval()

# Create a text classification pipeline
pipeline = transformers.pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)

# load dataset 

ds = load_dataset("Babelscape/ALERT", "alert", split="test")
df = ds.to_pandas()

# remove prefix and suffix
df.prompt = df.prompt.str.removeprefix("### Instruction:\n").str.removesuffix("\n### Response:\n")

categories = df["category"].unique()

results = {"category": [],
           "n_prompts": [],
           "n_false": []}


for category in categories:
    df_cat = df.loc[df['category'] == category]
    n = df_cat.shape[0]
    n_false = 0
    ground_truth = "LABEL_0" # -> harmful
    ds = Dataset.from_pandas(df_cat)
    for out in pipeline(KeyDataset(ds, "prompt"), batch_size=64):
        if out["label"] != ground_truth:
            n_false += 1
    results["category"].append(category)
    results["n_prompts"].append(n)
    results["n_false"].append(n_false)

    
with open('results.json', 'w') as fp:
    json.dump(results, fp, indent=2)

# for batch_size in [1, 8, 64, 256]:
#     print("-" * 30)
#     print(f"Streaming batch_size={batch_size}")
#     for out in tqdm(pipeline(KeyDataset(ds.select(range(256 * 4)), "prompt"), batch_size=batch_size)):
#         pass



