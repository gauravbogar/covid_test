import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

current_path = os.getcwd()
data_path = os.path.join(current_path, "assets", "covid.csv")
pickle_path = os.path.join(current_path, "assets", "covid.pkl")


def train(df, x, y):
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    cat_feat = [feature for feature in x.columns if df[feature].dtypes == "O"]
    num_feat = [feature for feature in x.columns if df[feature].dtypes != "O"]

    col_transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(drop="first"), cat_feat),
            ("ss", MinMaxScaler(), num_feat),
        ]
    )
    model = GradientBoostingClassifier()
    pipe = Pipeline(
        [("preprocessing", col_transformer), ("pca", PCA()), ("model", model)]
    )
    pipe.fit(x_train, y_train)
    # pickle.dump(pipe, open(pickle_path, 'wb'))
    # print('Pickle file created at ', pickle_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(pipe, f)
    return pickle_path


def test(pickle_path):
    # with open(pickle_path) as classifier:

    #     pickle.load(open(pickle_path), 'rb')
    classifier = pickle.load(open(pickle_path, "rb"))
    result = classifier.predict_proba(x_test.head(1))
    print(result)
    if result[0][1] < 0.25:
        print("Low Risk")
    elif (result[0][1] >= 0.25) and (result[0][1] < 0.5):
        print("Medium Risk")

    elif (result[0][1] >= 0.5) and (result[0][1] < 0.75):
        print("High Risk")

    else:
        print("Vulnerable Risk")


if __name__ == "__main__":

    df = pd.read_csv(data_path)
    x = df.drop("covid_res", axis=1)
    y = df["covid_res"]
    pickle_path = train(df, x, y)
    test(pickle_path)
