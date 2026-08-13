from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import os

# ALL_ATHLETES = {r["memberId"]: r["fullName"] for r in json.loads(Path("athlete_ids.json").read_text(encoding="utf-8"))}


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

def load_training_data(path= "training_data.csv"):
    return pd.read_csv(path)

def get_events(df):
    return (sorted(df["event"].unique()),
            sorted(df["distance"].unique()),
            sorted(df["stroke"].unique()),
            sorted(df["course"].unique()))

def load_event_df(df, event_name):
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

def fit_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "mae":  mean_absolute_error(y_test, predictions),
        "mse":  mean_squared_error(y_test, predictions),
        "r2":  r2_score(y_test, predictions)
    }

def compute_importance(model, X):
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': abs(model.coef_)
    })

    return importance.sort_values(by='Abs_Coefficient', ascending=False)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = fit_model(X_train, y_train)
    stats = evaluate_model(model, X_test, y_test)
    importance = compute_importance(model, X)

    return {
        "model": model,
        "stats": stats,
        "importance": importance,
        "numSamples": len(X),
        "df": X,
    }

def build_all_models(df, events):
    models = {}

    for event in events:

        X, y = load_event_df(df, event)

        if len(X) == 0:
            print(f"Skipping {event}. Not enough data to train model")
            continue

        models[event] = train_model(X, y)
        print(f"{event}: n={len(X)}")

    return models

def sort_events_by_sample_size(models):
    return sorted(
        models,
        key=lambda event: models[event]["numSamples"],
        reverse=True
    )

## Printing methods

def get_top_feature(importance_df):
    best_feature = importance_df.loc[importance_df["Abs_Coefficient"].idxmax()]
    return best_feature["Feature"]

def tally_top_features(models, events):
    features = {}  # track which feature is the most
    # print the training statistics
    for event in events:
        best_feature_name = get_top_feature(models[event]["importance"])
        features[best_feature_name] = features.get(best_feature_name, 0) + 1
    return features

def rank_features(features):
    """Returns dictionary of (feature, count) pairs sorted from most to least predictive by count"""
    return sorted(features.items(), key = lambda item: item[1], reverse=True)


def print_model_stats(event, model):
    stats = model["stats"]
    best_feature = get_top_feature(model["importance"])
    print(f"Printing training statisitcs for {event}")
    print(f"    MAE: {model["stats"]["mae"]:.3f} seconds")
    print(f"    MSE: {model["stats"]["mse"]:.3f} seconds")
    print(f"    R^2: {model["stats"]["r2"]:.3f}")
    print(f"    Most predictive feature: {best_feature}")


def print_feature_ranking(ranked, total):
    print("\n")
    max_feature, max_count = ranked[0]
    print(f"Most predictive feature for all models: {max_feature} ({max_count}/{total})")

    for i, (feature, count) in enumerate(ranked[1:], start=2):
        print(f"    #{i} most predictive feature for models: {feature} ({count}/{total})")


def feature_pie_plot(ranked, total, title):
    all_features = [item[0] for item in ranked]
    all_counts = [item[1] for item in ranked]
    plt.pie(all_counts, labels=all_features, autopct='%1.1f%%')
    plt.title(f'Most predictive features for {title} (n={total})')
    # plt.legend(title="Predictive Features")
    plt.savefig(f"figures/feature_pie_plot_{title.replace(" ", "_")}.png", dpi=300, bbox_inches="tight")
    plt.close()

def get_correlogram(models):
    for event in models:
        df = models[event]["df"]

        plt.figure(figsize=(12, 10), dpi=80)

        sns.heatmap(
            df.corr(),
            xticklabels=df.corr().columns,
            yticklabels=df.corr().columns,
            cmap="RdYlGn",
            center=0,
            annot=True
        )

        plt.title(f"Correlogram of variables for {event}", fontsize=18)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        plt.savefig(f"figures/correlogram_{event}.png", dpi=300, bbox_inches="tight")
        plt.close()


def print_report(models, events, title):
    print("\n")
    for event in events:
        print_model_stats(event, models[event])

    #tally which features are the top features, rank, then print
    features = tally_top_features(models, events)
    ranked = rank_features(features)
    print_feature_ranking(ranked, len(events))

    #show all graphics
    feature_pie_plot(ranked, len(events), title)
    get_correlogram(models)


def get_custom_data(models, events, distances, strokes, courses):
    print("Which course of events do you want information on (SCY, SCM, LCM)? Type N/A to skip. ", end="")
    course = input()
    while course not in courses and course != "n/a":
        print("Invalid course. Enter valid course or type N/A to skip. ", end="")
        course = input()

    print("Which stroke of events (FR, BK, BR, FL, IM)? Type N/A to skip. ", end="")
    stroke = input()
    while stroke not in strokes and stroke != "n/a":
        print("Invalid stroke. Enter valid stroke or type N/A to skip. ", end="")
        stroke = input()

    print("Which distance of events (50, 100, 200, 400, 500, 800, 1000, 1500, 1650) Type N/A to skip. ", end="")
    distance = input()
    while distance != "n/a" and int(distance) not in distances :
        print("Invalid distance (or not enough data on this distance to display)", end="")
        distance = input()

    filtered_models = {}
    filtered_events = []

    for event in events:

        # add all events of a specific course/stroke/distance, or all if N/A
        if ((course in event or course == "n/a")
                and (stroke in event or stroke == "n/a")
                and (distance in event or distance == "n/a")):

            filtered_events.append(event)

            if event in models:
                filtered_models[event] = models[event]

    if stroke == "n/a" and course == "n/a" and distance == "n/a":
        title = "all events"
    else:
        stroke_str = stroke if stroke != "n/a" else ""
        course_str = course if course != "n/a" else ""
        distance_str = (distance + "s" if distance != "n/a" else "")
        title = " ".join(
            part for part in [distance_str, stroke_str, course_str]
            if part
        )

    print(f"{title}: {filtered_events}")
    print("Are you alright with the selection of events? Enter 'no' to start over?")
    response = input()

    if response == 'no':
        get_custom_data(models, events, distances, strokes, courses)

    return filtered_models, filtered_events, title


def main():
    df = load_training_data()
    events, distances, strokes, courses = get_events(df)

    models = build_all_models(df, events)
    sorted_events = sort_events_by_sample_size(models)

    model_subset, events_subset, title = get_custom_data(models, sorted_events, distances, strokes, courses)

    print_report(model_subset, events_subset, title)

if __name__ == "__main__":
    main()
