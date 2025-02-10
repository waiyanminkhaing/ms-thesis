import pandas as pd
import ast

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

def safe_eval(val):
    """
    function to safe eval
    """
    return ast.literal_eval(val) if isinstance(val, str) else val