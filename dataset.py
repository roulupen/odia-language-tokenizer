"""
Dataset loader for OdiaGenAI dataset from Hugging Face.

This module loads the all_combined_odia_171k dataset from Hugging Face.
Authentication token can be provided via environment variable or .env file.
"""

import os
from datasets import load_dataset
from dotenv import load_dotenv


def load_odia_dataset(token=None, cache_dir="./data"):
    """
    Load the OdiaGenAI all_combined_odia_171k dataset from Hugging Face.
    
    Args:
        token (str, optional): Hugging Face authentication token. 
                              If not provided, will attempt to load from 
                              HUGGINGFACE_TOKEN environment variable.
        cache_dir (str, optional): Directory to cache the dataset. 
                                  Defaults to "./data".
    
    Returns:
        Dataset: The loaded dataset object.
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get token from parameter or environment variable
    if token is None:
        token = os.getenv('HUGGINGFACE_TOKEN')
    
    if token is None:
        print("Warning: No Hugging Face token provided. Loading may fail for private datasets.")
        print("To provide a token, either:")
        print("  1. Set HUGGINGFACE_TOKEN environment variable")
        print("  2. Create a .env file with HUGGINGFACE_TOKEN=your_token")
        print("  3. Pass token as argument to load_odia_dataset()")
    
    # Check if dataset is already cached
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        print(f"Loading dataset from cache directory: {cache_dir}")
    else:
        print(f"Downloading dataset and caching to: {cache_dir}")
    
    # Load the dataset with cache_dir parameter
    # print("Loading OdiaGenAI/all_combined_odia_171k dataset...")
    # ds = load_dataset(
    #     "OdiaGenAI/all_combined_odia_171k",
    #     token=token,
    #     cache_dir=cache_dir
    # )

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("OdiaGenAIdata/pre_train_odia_data_processed", cache_dir=cache_dir)
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset structure: {ds}")
    
    return ds


