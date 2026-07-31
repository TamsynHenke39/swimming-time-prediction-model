import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import os



ALL_ATHLETES = {r["memberId"]: r["fullName"] for r in json.loads(Path("athlete_ids.json").read_text(encoding="utf-8"))}

df = pd.read_csv('training_data.csv')

NUMERIC_FEATURES = [
        "ageAtSwim",
        "numPreviousSwims",
        "previousTime",
        "averageLast3",
        "bestPrevious",
        "stdPrevious",
        "daysSinceLast",
        "daysSinceBest",
        "ageAtBest",
        "yearsCompeting",
        "seasonCount",
    ]
CATEGORICAL_FEATURES = [
        "region",
        "isNcaa",
]
DROPPED_FEATURES =[
            "memberId",
            "fullName",
            "swimTime",   # target
            "event",
            "distance",
            "stroke",
            "meetName"
        ]

def load_event_df(event_name):
    event_df = df[df["event"] == event_name].copy()

    event_df= event_df.dropna() #drop rows without enough data points

    #one-hot encode categorical features
    event_df = pd.get_dummies(
        event_df,
        columns=CATEGORICAL_FEATURES,
        drop_first=True
    )

    # remove the dropped features
    X = event_df.drop(columns=DROPPED_FEATURES, errors="ignore")

    y = event_df['swimTime']
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    # Step 7 - make predictions
    y_pred = model.predict(X_test)
