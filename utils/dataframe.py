import os
import shutil
import tempfile
import time
import pandas as pd
import ast
import re
from datasets import load_from_disk, Dataset, concatenate_datasets

def save_gen_df(df, df_name):
    """
    function to save gen df
    """
    df.to_csv(f"gen/{df_name}.csv", index=False, encoding="utf-8")

def append_gen_df(df, df_name):
    """
    function to append gen df
    """
    df.to_csv(f"gen/{df_name}.csv", index=False, mode='a', header=False, encoding="utf-8")

def save_spt_df(df, df_name):
    """
    function to save spt df
    """
    df.to_csv(f"spt/{df_name}.csv", index=False, encoding="utf-8")

def load_gen_df(df_name):
    """
    function to load generated df
    """
    return pd.read_csv(f"gen/{df_name}.csv", header=0, encoding="utf-8")

def load_spt_df(df_name):
    """
    function to load spt df
    """
    return pd.read_csv(f"spt/{df_name}.csv", header=0, encoding="utf-8")

def save_models_df(df, df_name):
    """
    function to save models df
    """
    df.to_csv(f"models/{df_name}.csv", index=False, encoding="utf-8")

def save_tmp_df(df, df_name):
    """
    function to save tmp df
    """
    df.to_csv(f"tmp/{df_name}.csv", index=False, encoding="utf-8")

def load_spt_df(df_name):
    """
    function to load spt df
    """
    return pd.read_csv(f"spt/{df_name}.csv", header=0, encoding="utf-8")

def load_models_df(df_name):
    """
    function to load models df
    """
    return pd.read_csv(f"models/{df_name}.csv", header=0, encoding="utf-8")

def load_tmp_df(df_name):
    """
    function to load tmp df
    """
    return pd.read_csv(f"tmp/{df_name}.csv", header=0, encoding="utf-8")

def safe_eval(val):
    """
    function to safe eval
    """
    return ast.literal_eval(val) if isinstance(val, str) else val

def save_model_variants_df(df, df_name):
    """
    function to save model-variants df
    """
    df.to_csv(f"model-variants/data/{df_name}.csv", index=False, encoding="utf-8")

def save_model_variants_gen_df(df, df_name):
    """
    function to save model-variants df
    """
    df.to_csv(f"model-variants/gen/{df_name}.csv", index=False, encoding="utf-8")

def save_model_variants_hf(df, df_name, num_chunks=1): 
    """
    Function to save model-variants df in arrow format
    """
    output_dir = f"model-variants/data/{df_name}_hf_dataset"

    if num_chunks > 1:
        os.makedirs(output_dir, exist_ok=True)

        chunk_size = len(df) // num_chunks  # Compute chunk size dynamically
        remainder = len(df) % num_chunks  # Handle remaining samples

        start = 0
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)  # Distribute remainder
            chunk = df.select(range(start, end))
            chunk.save_to_disk(f"{output_dir}/chunk_{i}")
            start = end  # Move to next chunk
    else:
        df.save_to_disk(output_dir)

def save_model_variants_chunk_hf(df, df_name, chunk_num): 
    """
    Function to save model-variants df in arrow format
    """
    output_dir = f"model-variants/data/{df_name}_hf_dataset/chunk_{chunk_num}"

    # Ensure full cleanup before saving
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        time.sleep(1) 

    # Save to temp directory and move
    temp_dir = tempfile.mkdtemp()
    df.save_to_disk(temp_dir)

    shutil.move(temp_dir, output_dir)  # Move new dataset to original location

def load_model_variants_df(df_name):
    """
    function to load model-variants df
    """
    return pd.read_csv(f"model-variants/data/{df_name}.csv", header=0, encoding="utf-8")

def load_model_variants_gen_df(df_name):
    """
    function to load model-variants df
    """
    return pd.read_csv(f"model-variants/gen/{df_name}.csv", header=0, encoding="utf-8")

def natural_sort_key(path):
    """Extracts numeric chunk index from 'chunk_0', 'chunk_1', etc., for correct sorting."""
    match = re.search(r"chunk_(\d+)", path)  # Extracts the number
    return int(match.group(1)) if match else float('inf') 

def load_model_variants_hf(df_name, chunk_num=None):
    """
    Function to load model-variants df, handling both chunked and non-chunked datasets.
    """
    # Define the directory where the dataset is stored
    output_dir = f"model-variants/data/{df_name}_hf_dataset"

    # Check if dataset is chunked or a single dataset
    chunk_paths = sorted([
        os.path.join(output_dir, d) 
        for d in os.listdir(output_dir) 
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("chunk_")
    ], key=natural_sort_key)

    if chunk_paths:
        if chunk_num is not None:
            # if chunk_num is defined
            chunk_path = chunk_paths[chunk_num]
            print(chunk_path)
            full_dataset = load_from_disk(chunk_path)
        else:
            # If multiple subdirectories exist, assume dataset is chunked
            chunks = [load_from_disk(chunk_path) for chunk_path in chunk_paths]
            full_dataset = concatenate_datasets(chunks)
    else:
        # If no chunks are found, load as a single dataset
        full_dataset = load_from_disk(output_dir)

    return full_dataset

def convert_to_hf(dataset):
    """
    function to convert to huggingface dataset
    """
    return Dataset.from_pandas(dataset.reset_index(drop=True))