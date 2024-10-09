# Implements the version of the erase-and-check procedure that uses perplexity

import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, pipeline

import argparse, time, json, os
import numpy as np
from pathlib import Path

from defenses import progress_bar


def calculate_perplexity(output_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculate the perplexity for each sequence in a batch of sequences.

    Perplexity is a measurement of how well a probability distribution predicts a sample.
    In the context of language models, it's often used to evaluate the model's performance.
    Lower perplexity indicates better performance.

    Args:
        output_logits (torch.Tensor): The output logits from the model.
            Shape: (batch_size, sequence_length, vocab_size)
        input_ids (torch.Tensor): The input token IDs.
            Shape: (batch_size, sequence_length)

    Returns:
        torch.Tensor: The perplexity for each sequence in the batch.
            Shape: (batch_size,)

    Note:
        This function assumes that the model is using teacher forcing,
        where the input for predicting the next token is the ground truth
        from the previous time step.
    """
    batch_size = input_ids.shape[0]
    
    # Move input_ids to the same device as output_logits to enable model parallelism
    input_ids = input_ids.to(output_logits.device)
    
    # Shift logits and labels by one position
    # This aligns the predictions with the targets (next token prediction)
    shift_logits = output_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Calculate negative log-likelihood for each token
    loss_fct = CrossEntropyLoss(reduction="none")
    nll_tokenwise = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Reshape NLL values to match the batch structure
    nll_tokenwise = nll_tokenwise.view(batch_size, -1)
    
    # Calculate the average NLL for each sequence, which gives us the perplexity
    # Perplexity is e^(average NLL)
    perplexities = torch.exp(torch.mean(nll_tokenwise, dim=1))
    
    return perplexities


def create_2d_tensor_with_omissions(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a 2D tensor from a 1D input tensor where each row is the original tensor with one element omitted.

    This function takes a 1D PyTorch tensor and returns a 2D tensor. Each row of the output
    tensor is a copy of the input tensor with one element removed. The position of the
    removed element is different for each row, cycling through all possible positions.

    Args:
        input_tensor (torch.Tensor): A 1D PyTorch tensor.

    Returns:
        torch.Tensor: A 2D PyTorch tensor where each row is the input tensor with one element omitted.
                      The shape of the output tensor is (n, n-1), where n is the length of the input tensor.

    Raises:
        ValueError: If the input tensor is not 1-dimensional.

    Example:
        >>> input_tensor = torch.tensor([1, 2, 3, 4])
        >>> result = create_2d_tensor_with_omissions(input_tensor)
        >>> print(result)
        tensor([[2, 3, 4],
                [1, 3, 4],
                [1, 2, 4],
                [1, 2, 3]])
    """
    if input_tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")

    # Get the length of the input tensor
    length = input_tensor.size(0)
    
    # Create a range tensor [0, 1, 2, ..., length-1]
    indices = torch.arange(length)
    
    # Create a mask for each row
    mask = (indices.unsqueeze(0) != indices.unsqueeze(1))
    
    # Use the mask to select elements for each row
    result = input_tensor.unsqueeze(0).expand(length, -1)[mask].view(length, -1)
    
    return result


def perplexity_ec(prompt, classifier_pipe, label_to_classification, perplexity_model, perplexity_tokenizer, device, num_iters, output_subsequence=False):
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
    tokenized_prompt = perplexity_tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    
    # Otherwise, erase tokens until the prompt is labelled harmful
    for _ in range(num_iters):
        # Create batch of inputs with one token erased
        tokenized_subsequences_batch = create_2d_tensor_with_omissions(tokenized_prompt)

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
        if tokenized_subsequences_batch.shape[1] == 2:
            if output_subsequence:
                return False, "prompt was reduced to 2 tokens without being detected harmful"
            return False
        
        # find subsequence with the lowest perplexity
        with torch.no_grad():
            output = perplexity_model(tokenized_subsequences_batch)
            logits = output.logits
            perplexities = calculate_perplexity(logits, tokenized_subsequences_batch)
        
        # keep subsequence with lowest perplexity for next iteration
        argmin = torch.argmin(perplexities)
        tokenized_prompt = tokenized_subsequences_batch[argmin]

    if output_subsequence:
        return False, "maximum number of iterations was reached without any subsequence being detected harmful"
    return False


def random_token_erasing_ec(prompt, classifier_pipe, label_to_classification, num_iters, seed, output_subsequence=False):
    rng = np.random.default_rng(seed=seed)
    # Check the whole prompt with the safety classifier
    classifier_output = classifier_pipe(prompt)[0]
    output_class = label_to_classification[classifier_output["label"]]

    # If the prompt is initially labelled harmful, return True
    # Class 1: Safe, Class 0: Harmful
    if output_class == 0:
        if output_subsequence:
            return True, prompt
        return True
    
    # split the prompt
    tokens = prompt.split()
    
    # Otherwise, erase tokens until the prompt is labelled harmful
    for _ in range(num_iters):
        n_tokens = len(tokens)
        if n_tokens == 1:
            if output_subsequence:
                return False, tokens.pop()
            else:
                return False 
        random_idx = rng.integers(0, n_tokens).item()
        tokens.pop(random_idx)
        
        # check remaining prompt
        prompt = " ".join(tokens)
        classifier_output = classifier_pipe(prompt)[0]
        output_class = label_to_classification[classifier_output["label"]]
        
        # Class 1: Safe, Class 0: Harmful
        if output_class == 0:
            if output_subsequence:
                return True, prompt
            return True

    if output_subsequence:
        return False, prompt
    return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, required=True, help='File containing prompts')
    parser.add_argument("--num_adv", type=int, required=True, help="Number of adversarial tokens")
    parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations')
    parser.add_argument('--results_file', type=str, default='results/perplexity_ec_results.json', help='File to store results')
    parser.add_argument("--device", "-d", type=str, choices=["mps", "cuda"])
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--perplexity_model", "-p", type=str, choices=["gpt2", "tinyllama", "random"])
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)

    # load perplexity model
    if args.perplexity_model == "gpt2":
        model_id = "openai-community/gpt2-large"
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        model.eval()
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    elif args.perplexity_model == "tinyllama":
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()
    else:
        model_id = "random"
        model = None
        tokenizer = None
    
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
    num_adv = args.num_adv
    num_iters = args.num_iters
    results_file = args.results_file

    print('\n* * * * * * * Experiment Details * * * * * * *')
    print('Prompts file:\t', prompts_file)
    print('Iterations:\t', str(num_iters))
    print("Number of adversarial tokens:\t", str(num_adv))
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
    results_dict = {"perplexity_model": model_id}
    # Create results file if it does not exist
    if not os.path.exists(results_file):
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
    else:
        with open(results_file, 'r') as f:
            results_dict = json.load(f)

    for num_done, input_prompt in enumerate(prompts):
        if model:
            decision, subsequence = perplexity_ec(input_prompt, clf_pipe, label_to_class, model, tokenizer, device, num_iters, output_subsequence=True)
        else:
            decision, subsequence = random_token_erasing_ec(input_prompt, clf_pipe, label_to_class, num_iters, args.seed, True)
        list_of_bools.append(decision)
        list_of_subsequences.append(subsequence)
        percent_harmful = (sum(list_of_bools) / len(list_of_bools)) * 100.
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (num_done + 1)

        print("  Checking safety... " + progress_bar((num_done + 1) / len(prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r")
        
    # for bool, subsequence in zip(list_of_bools, list_of_subsequences):
    #     print(bool, subsequence)
    print("")

    # Save results
    if not (str(num_iters) in results_dict.keys()):
        results_dict[str(num_iters)] = {}
    results_dict[str(num_iters)][str(num_adv)] = dict(percent_harmful = percent_harmful, time_per_prompt = time_per_prompt)
    print("Saving results to", results_file)
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
