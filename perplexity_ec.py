# Implements the version of the erase-and-check procedure that uses perplexity

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, pipeline

import argparse, time, json, os
import numpy as np
from pathlib import Path

from defenses import progress_bar

def perplexity_ec(prompt, classifier_pipe, label_to_classification, perplexity_model, perplexity_tokenizer, device, num_iters=20, output_subsequence=False):
    '''
    Implements the perplexity version of the erase-and-check procedure
    At each step, the algorithm erases a token from the current prompt
    such that the remaining prompt minimizes the perplexity 
    (exponentiated average negative log-likelihood). It repeats
    this process until the safety classifier labels the prompt as harmful. 
    If the model does not label the prompt as harmful, the algorithm declares
    the original prompt as safe.

    Args:
        prompt: Input prompt (string)
        classifier_pipe: pipeline of the prompt classifier
        label_to_classification: dict that converts the output from the classifier pipe into 0 for harmful or 1 otherwise
        perplexity_model: autoregressive model used for calculating perplexity of token sequence
        perplexity_tokenizer: tokenizer for the perplexity model
        device: gpu device
        num_iters: Number of iterations

    Returns:
        True if the prompt is harmful, False otherwise
    '''

    

    # Check the whole prompt with the safety classifier
    classifier_output = classifier_pipe(prompt)[0]
    output_class = label_to_classification[classifier_output["label"]]

    # If the prompt is initially labelled harmful, return True
    # Class 1: Safe, Class 0: Harmful
    if output_class == 0:
        if output_subsequence:
            return True, prompt
        return True
    
    # Tokenize the prompt
    tokenized_prompt = perplexity_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Otherwise, erase tokens until the prompt is labelled harmful
    for _ in range(num_iters):
        # Create batch of inputs with one token erased
        tokenized_subsequences_batch = []
        n_tokens = tokenized_prompt.shape[1]
        for i in range(n_tokens):
            tokenized_subsequences_batch.append(tokenized_prompt[0, np.r_[0:i, i+1:n_tokens]])
            
        tokenized_subsequences_batch = torch.stack(tokenized_subsequences_batch, dim=0)

        # Check each subsequence with the classifier pipeline
        classifier_outputs = classifier_pipe(perplexity_tokenizer.batch_decode(tokenized_subsequences_batch))
        output_classes = [label_to_classification[output["label"]] for output in classifier_outputs]
        for i, output_class in enumerate(output_classes):
            if output_class == 0:
                if output_subsequence:
                    return True, perplexity_tokenizer.decode(tokenized_subsequences_batch[i])
                else:
                    return True
        
        # the perplexity of a sequence with one token cannot be calculated, therefore return False (not harmful) when only 2 tokens are left
        if n_tokens == 2:
            if output_subsequence:
                return False, perplexity_tokenizer.decode(tokenized_prompt.squeeze())
            return False
        
        # find subsequence with the lowest perplexity
        perplexities = []
        for tokenized_subsequence in tokenized_subsequences_batch:
            # when passing the model the input ids also as labels, the loss will be the perplexity
            perplexity = perplexity_model(tokenized_subsequence, labels=tokenized_subsequence.clone()).loss.item()
            perplexities.append(perplexity)
        
        # extract subsequence with lowest perplexity for next iteration
        tokenized_prompt = tokenized_subsequences_batch[np.argmin(perplexities)].unsqueeze(0)

    if output_subsequence:
        return False, prompt
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, default='data/adversarial_prompts_t_10.txt', help='File containing prompts')
    parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--results_file', type=str, default='results/perplexity_ec_results.json', help='File to store results')
    parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])
    parser.add_argument("--model_path", "-m", type=str, required=True)

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)

    # load perplexity model
    model_id = "openai-community/gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    
    # load classifier model
    
    if "distilbert" in args.model_path:
        # Load the tokenizer
        clf_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # pass the pre-trained DistilBert to our define architecture
        clf_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    else:
        # Load the tokenizer
        clf_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')

        # pass the pre-trained DistilBert to our define architecture
        clf_model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)

    clf_model.load_state_dict(torch.load(args.model_path, map_location=device))
    clf_model.eval()
    clf_pipe = pipeline('text-classification', model=clf_model, tokenizer=clf_tokenizer, device=device)
    label_to_class = {
        "LABEL_0": 0,
        "LABEL_1": 1
    }

    prompts_file = args.prompts_file
    num_iters = args.num_iters
    results_file = args.results_file

    print('\n* * * * * * * Experiment Details * * * * * * *')
    print('Prompts file:\t', prompts_file)
    print('Iterations:\t', str(num_iters))
    print('* * * * * * * * * * * ** * * * * * * * * * * *\n')

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")
    list_of_bools = []
    list_of_subsequences = []
    start_time = time.time()

    # Open results file and load previous results JSON as a dictionary
    results_dict = {}
    # Create results file if it does not exist
    if not os.path.exists(results_file):
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w+') as f:
            json.dump(results_dict, f)
    with open(results_file, 'r') as f:
        results_dict = json.load(f)

    for num_done, input_prompt in enumerate(prompts):
        decision, subsequence = perplexity_ec(input_prompt, clf_pipe, label_to_class, model, tokenizer, device, num_iters, output_subsequence=True)
        list_of_bools.append(decision)
        list_of_subsequences.append(subsequence)
        percent_harmful = (sum(list_of_bools) / len(list_of_bools)) * 100.
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (num_done + 1)

        print("  Checking safety... " + progress_bar((num_done + 1) / len(prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r")
        
    for bool, subsequence in zip(list_of_bools, list_of_subsequences):
        print(bool, subsequence)
    print("")

    # Save results
    results_dict[str(dict(num_iters = num_iters))] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt)
    print("Saving results to", results_file)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
