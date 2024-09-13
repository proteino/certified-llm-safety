from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
import argparse
from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"], required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--batch_size", "-b", type=int, required=True)
parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_filename", type=str, default="distilbert.pt")
parser.add_argument("--patience", type=int)
parser.add_argument("--n_harmful", type=int)
parser.add_argument("--n_safe", type=int)
parser.add_argument("--max_len", type=int, required=True)
parser.add_argument("--progress_bar", action="store_true")
parser.add_argument("--bert_version", type=str, choices=["distilbert", "bert"], default="distilbert")
parser.add_argument("--use_ethical_decision_making_dataset", "-e", action="store_true")




args = parser.parse_args()
device = torch.device(args.device)
model_save_path = "models/" + args.save_filename
patience = args.patience
seed = args.seed
num_epochs = args.num_epochs
learning_rate = args.learning_rate
max_token_length = args.max_len
batch_size = args.batch_size
train_test_split = [0.9, 0.1]
torch.manual_seed(seed)


# load harmful prompts dataset
harmful_prompts_dataset = load_dataset("Babelscape/ALERT", "alert", split="test") # ignore the split, dataset only has test split

# preprocess harmful prompts dataset

# downsize if necessary
if args.n_harmful and args.n_harmful < harmful_prompts_dataset.shape[0]:
    harmful_prompts_dataset = harmful_prompts_dataset.shuffle(seed=seed).select(range(args.n_harmful))

n_harmful = harmful_prompts_dataset.shape[0]


# remove prefix and suffix
data_frame = harmful_prompts_dataset.to_pandas()
data_frame.prompt = data_frame.prompt.str.removeprefix("### Instruction:\n").str.removesuffix("\n### Response:\n")

# turn categories into labels
categories = data_frame.category.unique()
category_to_label = {text: i + 1 for i, text in enumerate(categories)}
label_to_category = {i+1: text for i, text in enumerate(categories)}
num_classes = len(categories) + 1
data_frame["labels"] = data_frame["category"].map(category_to_label)
harmful_prompts_dataset = Dataset.from_pandas(data_frame)
del data_frame

#remove unused columns
harmful_prompts_dataset = harmful_prompts_dataset.remove_columns(["id", "category"])


# load safe prompts dataset
safe_prompts_dataset = load_dataset("THUDM/webglm-qa", split="train")

#remove unused columns and rename column
safe_prompts_dataset = safe_prompts_dataset.remove_columns(["answer", "references"]).rename_column("question", "prompt")

if args.use_ethical_decision_making_dataset:

    # load self made ethical decision dataset
    ethical_decision_making_ds = load_dataset("grossjct/ethical_decision_making_prompts", split="train")
    ethical_decision_making_ds = ethical_decision_making_ds.remove_columns(["id"])

    # fuse safe prompt datasets
    safe_prompts_dataset = concatenate_datasets([safe_prompts_dataset, ethical_decision_making_ds])


# downsize if necessary
if args.n_safe and args.n_safe < safe_prompts_dataset.shape[0]:
    safe_prompts_dataset = safe_prompts_dataset.shuffle(seed=seed).select(range(args.n_safe))

n_safe = safe_prompts_dataset.shape[0]


# add label column, 0 -> safe
safe_prompts_dataset = safe_prompts_dataset.add_column("labels", [0] * safe_prompts_dataset.shape[0])


#fuse datasets and shuffle
fused_dataset = concatenate_datasets([harmful_prompts_dataset, safe_prompts_dataset]).shuffle(seed=seed)
del safe_prompts_dataset, harmful_prompts_dataset

if args.bert_version == "distilbert":

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # pass the pre-trained DistilBert to our define architecture
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
else:
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

    # pass the pre-trained DistilBert to our define architecture
    model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=num_classes)
    
# tokenize dataset
fused_dataset = fused_dataset.map(lambda datapoint: tokenizer(datapoint["prompt"], max_length=max_token_length, padding="max_length", truncation=True), batched=True)

# remove prompts column
fused_dataset = fused_dataset.remove_columns(["prompt"])

# split dataset
fused_dataset = fused_dataset.train_test_split(train_size=train_test_split[0], test_size=train_test_split[1], shuffle=True, seed=seed)

# set data format to torch
fused_dataset.set_format("torch")

# create torch data loaders
train_dataloader = DataLoader(fused_dataset["train"], batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(fused_dataset["test"], batch_size=batch_size)

# move model to device
model.to(device)

num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

optimizer = AdamW(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
eval_losses = []
eval_accuracies = []
best_eval_loss = torch.inf
n_stagnating_epochs = 0

multiclass_accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1} / {num_epochs}')
    print("\nTraining...")

    # training
    model.train()
    total_train_loss = 0
    batch_accuracies = []
    # progress_bar = tqdm(range(len(train_dataloader)))
    for step, batch in enumerate(train_dataloader):
        # move batch to gpu
        batch = {k: v.to(device) for k, v in batch.items()}
        
        
        # generate predictions
        outputs = model(**batch)
        
        # generate loss and do backpropagation
        loss = outputs.loss
        loss.backward()
        
        total_train_loss += loss.item()

        # optimize weights
        optimizer.step()
        optimizer.zero_grad()
        
        # compute accuracy
        preds = torch.argmax(outputs.logits, dim=-1)
        batch_accuracies.append(multiclass_accuracy(batch["labels"], preds).item())
        
        if args.progress_bar:
            progress_bar.update(1)
    
    train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(train_loss)
    train_accuracy = np.mean(batch_accuracies).item()
    train_accuracies.append(train_accuracy)
    
    # evaluation
    model.eval()
    print("\nEvaluating...")
    # progress_bar = tqdm(range(len(test_dataloader)))
    total_eval_loss = 0
    batch_accuracies = []
    for step, batch in enumerate(test_dataloader):
        # move batch to gpu
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        
        total_eval_loss += loss.item()
        
        # compute accuracy
        preds = torch.argmax(outputs.logits, dim=-1)
        batch_accuracies.append(multiclass_accuracy(batch["labels"], preds).item())
        
        
    
    eval_loss = total_eval_loss / len(test_dataloader)
    eval_losses.append(eval_loss)
    eval_accuracy = np.mean(batch_accuracies).item()
    eval_accuracies.append(eval_accuracy)
    
    
    # log losses
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Training Accuracy: {train_accuracy:.3f}')
    print(f'Eval Loss: {eval_loss:.3f}')
    print(f'Eval Accuracy: {eval_accuracy:.3f}')
    print(f'Best eval Loss: {min(best_eval_loss, eval_loss):.3f}')
    
    # save best model
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        n_stagnating_epochs = 0
    elif patience:
        # early stopping
        n_stagnating_epochs += 1
        if n_stagnating_epochs >= patience:
            print(f"Stopping early in epoch {epoch} because the training stagnated for {n_stagnating_epochs} epochs")
            break
        
    

log = {}
log["train_losses"] = train_losses
log["train_accuracies"] = train_accuracies
log["eval_losses"] = eval_losses
log["eval_accuracies"] = eval_accuracies

hyperparams = {
    "max_token_length": max_token_length,
    "lr": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "n_harmful": n_harmful,
    "n_safe": n_safe,
    "split": train_test_split
}

log["hyperparams"] = hyperparams

with open(f'models/{args.save_filename.removesuffix(".pt")}_loss_log.json', 'w') as fp:
    json.dump(log, fp, indent=2)
    
        
        
        
        
        









