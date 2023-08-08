import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os,ast
from scripts.tf_utils import compute_cosine_sim as tf_cosine_sim
from scripts.tf_utils import convert_to_tensor as tf_convert_to_tensor

from scripts.torch_utils import compute_cosine_sim as torch_cosine_sim
from scripts.torch_utils import convert_to_tensor as torch_convert_to_tensor


DATA_FILE_PATH = os.getcwd() + "/../data/Zomato_cleaned.csv"
MODELS = ["all-MiniLM-L6-v2"]
OPTIONS = {1: "torch", 2: "tf"}


def compute_text_embeddings(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = SentenceTransformer(model_name)

    features_to_encode = df["combined_text"].values
    embeddings = model.encode(features_to_encode)

    df["embeddings"] = embeddings.tolist()
    df["embeddings"] = df["embeddings"].to_numpy()

    file_name = f"{model_name}_zomato_embeddings.csv"
    file_path = f"{os.getcwd()}/../data/embeddings/{file_name}"

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
        embeddings_df = load_embeddings(str(file_path))
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


def recommend(query_name: str, df: pd.DataFrame, model: str, option: str = "torch"):
    df["combined_text"] = df["cuisine"] + " " + df["timing"] + " " + str(df["cost"]) + " " + str(df["rating"])
    embeddings_df = get_text_embeddings(df[["name", "combined_text"]],model)

    query_cuisine = df.loc[df["name"] == query_name]["cuisine"].values[0]
    print(f"recommendations similar to {query_name} of {query_cuisine} cuisine are as follows \n")
    df_remaining = df.loc[df["name"] != query_name]

    query_embeddings = embeddings_df.loc[embeddings_df["name"] == query_name]["embeddings"].values[0]
    remaining_embeddings = embeddings_df.loc[embeddings_df["name"] != query_name]["embeddings"].values

    # converting them from numpy ndarray to tensors
    if option == "torch":
        query_embeddings, remaining_embeddings = torch_convert_to_tensor(query_embeddings, remaining_embeddings)
        results = torch_cosine_sim(query_embeddings, remaining_embeddings)
    else:
        query_embeddings, remaining_embeddings = tf_convert_to_tensor(query_embeddings, remaining_embeddings)
        results = tf_cosine_sim(query_embeddings, remaining_embeddings)
    df_remaining.loc[:, "sim_scores"] = results
    df_remaining = df_remaining.sort_values(by=["sim_scores"], ascending=False)

    return df_remaining.reset_index(drop=True)


def main():
    OPTION = 2
    data = pd.read_csv(DATA_FILE_PATH, sep=",")
    recommendations = recommend("New Arsalaan Biryani", data, MODELS[0], OPTIONS[OPTION])
    print(recommendations[["name", "cuisine", "rating", "sim_scores"]].head())


if __name__ == '__main__':
    main()
