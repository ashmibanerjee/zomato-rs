import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os, ast
from typing import List
import torch

DATA_FILE_PATH = os.getcwd() + "/../data/Zomato_cleaned.csv"
MODEL = "all-MiniLM-L6-v2"
# FEATURE_VECTOR = ["voteCount", "rating", "cost", "combined_text"]
FEATURE_VECTOR = ["combined_text"]


def compute_text_embeddings(df: pd.DataFrame, model_name: str, features: List) -> pd.DataFrame:
    names = df["name"]
    df["combined_text"] = df["cuisine"] + " " + df["timing"]
    df.drop(columns=["cuisine", "timing"], inplace=True)
    model = SentenceTransformer(model_name)

    print("inside compute embeddings: ", df.shape)
    features_to_encode = df["combined_text"].values
    print(df.shape)
    print(features_to_encode)

    embeddings = model.encode(features_to_encode)

    df["embeddings"] = embeddings.tolist()
    df["embeddings"] = df["embeddings"].to_numpy()

    file_name = f"{model_name}_zomato_embeddings.csv"
    file_path = f"{os.getcwd()}/../data/embeddings/{file_name}"
    print(file_path)
    df[["name", "embeddings"]].to_csv(file_path, index=False)

    print("embeddings computed")
    return df


def load_embeddings(file_path: str) -> pd.DataFrame:
    embeddings_df = pd.read_csv(file_path)
    return embeddings_df


def get_text_embeddings(df: pd.DataFrame, model_name: str, features: List) -> pd.DataFrame:
    file_name = f"{model_name}_zomato_embeddings.csv"
    file_path = Path(f"{os.getcwd()}/../data/embeddings/{file_name}")

    if file_path.is_file():
        embeddings_df = load_embeddings(file_path)
    else:
        embeddings_df = compute_text_embeddings(df, model_name, features)
    try:
        # transforming them from str to numpy ndarray
        embeddings_df["embeddings"] = embeddings_df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    except ValueError as e:
        print(e)
    return embeddings_df


def rescale_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_str = df.select_dtypes(include=['object'])
    df_to_scale = df.select_dtypes(include=['int64', 'float64'])

    scaler = MinMaxScaler()

    df_scaled = pd.DataFrame(scaler.fit_transform(df_to_scale), columns=df_to_scale.columns)
    # Combine the scaled columns with the 'name' column
    df_scaled = pd.concat([df_str, df_scaled], axis=1)

    return df_scaled


def compute_cosine_sim(query_vector, remaining_vector):
    results = util.cos_sim(query_vector, remaining_vector)
    return results.flatten().tolist()


def convert_to_tensor(query_vals, remaining_vals):
    if type(query_vals) == list:
        query_embeddings = torch.tensor(query_vals)
    else:
        query_embeddings = torch.from_numpy(query_vals)
    remaining_embeddings = np.vstack(remaining_vals).astype(float)
    remaining_embeddings = torch.from_numpy(remaining_embeddings)
    return query_embeddings, remaining_embeddings


def recommend(query_name: str, df: pd.DataFrame):

    embeddings_df = get_text_embeddings(df, MODEL, FEATURE_VECTOR)

    query_cuisine = df.loc[df["name"] == query_name]["cuisine"].values[0]
    print(f"recommendations similar to {query_name} of {query_cuisine} cuisine are as follows \n")
    df_remaining = df.loc[df["name"] != query_name]
    # df_scaled = rescale_numeric_columns(df_remaining)

    query_embeddings = embeddings_df.loc[embeddings_df["name"] == query_name]["embeddings"].values[0]
    remaining_embeddings = embeddings_df.loc[embeddings_df["name"] != query_name]["embeddings"].values

    # converting them from numpy ndarray to tensors
    query_embeddings, remaining_embeddings = convert_to_tensor(query_embeddings, remaining_embeddings)

    results = compute_cosine_sim(query_embeddings, remaining_embeddings)
    df_remaining.loc[:, "sim_scores"] = results
    df_remaining = df_remaining.sort_values(by=["sim_scores"], ascending=False)

    return df_remaining.reset_index(drop=True)


def main():
    data = pd.read_csv(DATA_FILE_PATH, sep=",")
    recommendations = recommend("Super Star Haji Biriyani", data)
    print(recommendations[["name", "cuisine", "sim_scores"]].head())


if __name__ == '__main__':
    main()
