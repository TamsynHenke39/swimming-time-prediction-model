from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        "isNcaa",
]
DROPPED_FEATURES =[
            "memberId",
            "fullName",
            "swimTime",   # target
            "event",
            "distance",
            "stroke",
            "course",
            "meetName",
            "swimDate",
            "region",
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

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # print(f"    MAE: {mae:.3f} seconds")
    # print(f"    MSE: {mse:.3f} seconds")
    # print(f"    R^2: {r2:.3f}

    stats = {"mae": mae,"mse": mse, "r2": r2}

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': abs(model.coef_)
    })

    importance = importance.sort_values(by='Abs_Coefficient', ascending=False)

    model_dict = {"model": model, "stats": stats, "importance": importance, "numSamples": len(X)}
    return model_dict

def main():

    models = {}
    events = sorted(df["event"].unique())

    for event in events:

        X, y = load_event_df(event)

        if len(X) == 0:
            print(f"Skipping {event}. Not enough data to train model")
            continue

        models[event] = train_model(X, y)
        print(f"{event}: {len(X)} samples")

    sorted_events = sorted(
        models,
        key=lambda event: models[event]["numSamples"],
        reverse=True
    )

    print("\n")
    #print the training statistics
    for event in sorted_events:
        model = models[event]
        print(f"Printing training statisitcs for {event}")
        print(f"    MAE: {model["stats"]["mae"]:.3f} seconds")
        print(f"    MSE: {model["stats"]["mse"]:.3f} seconds")
        print(f"    R^2: {model["stats"]["r2"]:.3f}")

        best_feature = model["importance"].loc[model["importance"]["Abs_Coefficient"].idxmax()]
        print("Most predictive feature:", best_feature["Feature"])

    #print the coefficeints that are the most important

    #print the

if __name__ == "__main__":
    main()
