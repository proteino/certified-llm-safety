#!/bin/sh

echo "creating and activating venv"
VENV_NAME="venv"
# Create a virtual environment
python3 -m venv $VENV_NAME

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip
pip install --upgrade pip


# Install requirements
pip install torch transformers

echo "Virtual environment '$VENV_NAME' has been created and requirements have been installed."
echo "The virtual environment is now activated."


./bash_scripts/jobs_perplexity_ec.sh cuda /kaggle/input/distilbert_infusion/pytorch/authors/2/benchmark_distilbert_suffix.pt tinyllama

rm -rf $VENV_NAME

