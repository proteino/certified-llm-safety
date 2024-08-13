import pandas as pd

df = pd.read_json("hf://datasets/Babelscape/ALERT/alert.jsonl", lines=True)

def column_to_string(df, column_name):
    """
    Convert a specified column of a pandas DataFrame to a string.
    Each line of the string corresponds to a row in the DataFrame.
    
    Args:
    df (pandas.DataFrame): The input DataFrame
    column_name (str): The name of the column to convert
    
    Returns:
    str: A string representation of the specified column
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    
    # Convert the column to a list of strings and join with newlines
    return '\n'.join(df[column_name].astype(str).tolist())

df.prompt = df.prompt.str.removeprefix("### Instruction:\n").str.removesuffix("\n### Response:\n")
prompts_string = column_to_string(df, "prompt")
with open("alert_dataset_harmful_prompts.txt", "w") as text_file:
    text_file.write(prompts_string)

