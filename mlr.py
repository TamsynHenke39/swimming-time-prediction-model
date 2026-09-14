from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def load_training_data(path= "scraping/training_data.csv"):
    """
    Loads training data into Dataframe from CSV

    Args:
        path - the location of the CSV

    Returns:
        the Dataframe read from the CSV
    """
    return pd.read_csv(path)

def get_events(df):
    """
    Creates a dataframe of all unique events, strokes, distances, and courses

    Args:
        df: the Dataframe loaded from the training CSV

    Returns:
        a data frame sorted by unique, event, distance, stroke, and course
    """
    return (sorted(df["event"].unique()),
            sorted(df["distance"].unique()),
            sorted(df["stroke"].unique()),
            sorted(df["course"].unique()))

def load_event_df(df, event_name):
    """
    Filters training dataframe to a single event, splits into features and target
    Filters df to rows matching event_name, drops rows with missing values, one-hot encodes
    categorical features, and removes columns not used for training

    Args:
        df: the Dataframe loaded from the training CSV
        event_name: full name of the event, ex: "100 FR SCY"

    Returns:
        a tuple (X, y) where X is the filtered, encoded feature DataFrame and y is the target swim times

    """
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
    """
    Fits multilinear regression model to the training data

    Args:
        X_train: the feature DataFrame for a given event
        y_train: the target swim times corresponding to X_train

    Returns:
         fitted multilinear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates multilinear regression model on test set using mean absolute error (MAE),
    mean squared error (MSE), and R^2

    Args:
        model: the multilinear regression model for a given event
        X_test: test set of features for a given event
        y_test: corresponding test set of target swim times

    Returns:
        dictionary of model performance using MAE, MSE, & R2 metrics

    """
    predictions = model.predict(X_test)
    return {
        "mae":  mean_absolute_error(y_test, predictions),
        "mse":  mean_squared_error(y_test, predictions),
        "r2":  r2_score(y_test, predictions)
    }

def compute_importance(model, X):
    """
    Creates dataframe sorted by "importance" of MLR model's features,  the absolute
    value of the feature coefficient

    Args:
        model: the multilinear regression model for a given event
        X: given features of the model

    Returns:
        Dataframe of model features, sorted by absolute value of coefficients
    """
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': abs(model.coef_)
    })

    return importance.sort_values(by='Abs_Coefficient', ascending=False)

def train_model(X, y):

    """
    Trains multilinear regression model for a given event & computes its performance statistics and feature importances

    Args:
        X: feature dataframe for a given event
        y: associated target swim times

    Returns:
        a dict aggregating the model's info: the fitted model, performance stats (MAE, MSE, R^2), feature importance
        DataFrame, number of training samples, and the feature DataFrame used
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #sc = StandardScaler()

    #X_train_scaled = sc.fit_transform(X_train)
    #X_test_scaled = sc.transform(X_test)

    #pca = PCA(n_components=0.95)

    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)
    #
    # explained_variance = pca.explained_variance_ratio_

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
    """
    Trains a model for every event with enough data

    Args:
        df: the Dataframe loaded from the training CSV
        events: the event names to train models for

    Returns:
        a dictionary mapping event names to that event's model
    """
    models = {}

    for event in events:

        X, y = load_event_df(df, event)

        if len(X) == 0:
            print(f"Skipping {event}. Not enough data to train model")
            continue

        models[event] = train_model(X, y)
        print(f"{event}: n={len(X)}")

    return models