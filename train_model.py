import pandas as pd
from sklearn.cluster import KMeans
import pickle


def main():
    # Path to the Kaggle dataset downloaded separately
    data_path = "data/data-final.csv"
    df_raw = pd.read_csv(data_path, sep='\t')
    data = df_raw.copy()
    # remove additional questionnaire columns
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)

    # drop country column for modelling
    df_model = data.drop('country', axis=1)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_model)

    with open('trained_Model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()
