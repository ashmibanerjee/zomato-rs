import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os,ast
import torch

DATA_FILE_PATH = os.getcwd() + "/../data/Zomato_cleaned.csv"
MODEL = "all-MiniLM-L6-v2"


def compute_text_embeddings(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = SentenceTransformer(model_name)

    print("inside compute embeddings: ", df.shape)
    features_to_encode = df["combined_text"].values
    print(df.shape)

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


def get_text_embeddings(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    file_name = f"{model_name}_zomato_embeddings.csv"
    file_path = Path(f"{os.getcwd()}/../data/embeddings/{file_name}")

    if file_path.is_file():
        embeddings_df = load_embeddings(file_path)
        embeddings_df["embeddings"] = embeddings_df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    else:
        embeddings_df = compute_text_embeddings(df, model_name)
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

    if type(query_vals) is list:
        query_embeddings = torch.FloatTensor(query_vals).float()
    else:
        query_embeddings = torch.from_numpy(query_vals).float()
    remaining_embeddings = np.vstack(remaining_vals).astype(float)
    remaining_embeddings = torch.from_numpy(remaining_embeddings).float()
    return query_embeddings, remaining_embeddings


def recommend(query_name: str, df: pd.DataFrame):
    df["combined_text"] = df["cuisine"] + " " + df["timing"] + " " + str(df["cost"]) + " " + str(df["rating"])
    embeddings_df = get_text_embeddings(df[["name", "combined_text"]], MODEL)

    query_cuisine = df.loc[df["name"] == query_name]["cuisine"].values[0]
    print(f"recommendations similar to {query_name} of {query_cuisine} cuisine are as follows \n")
    df_remaining = df.loc[df["name"] != query_name]

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
    recommendations = recommend("New Arsalaan Biryani", data)
    print(recommendations[["name", "cuisine", "rating", "sim_scores"]].head())


if __name__ == '__main__':
    main()
