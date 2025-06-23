import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
import pickle


def main():
    parser = argparse.ArgumentParser(description="Train the personality clustering model")
    parser.add_argument(
        "--data-path",
        default="data/data-final.csv",
        help="Path to the Kaggle dataset",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        parser.error(f"Data file '{data_path}' not found")

    df_raw = pd.read_csv(data_path, sep='\t')
    data = df_raw.copy()
    # remove additional questionnaire columns
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)

    # drop country column for modelling
    df_model = data.drop('country', axis=1)

    # remove rows with missing values to avoid NaNs during training
    df_model.dropna(inplace=True)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_model)

    with open('trained_Model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()
