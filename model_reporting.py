import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


## Printing methods

def sort_events_by_r2(models):
    """
    Sorts event names by their model's R^2 score, best-fitting first

    Args:
        models: a dictionary mapping event names to model information

    Returns:
        a list of event names sorted by R^2, in descending order
    """
    return sorted(
        models,
        key=lambda event: models[event]["stats"]["r2"],
        reverse=True
    )


def get_top_feature(importance_df):
    """
    Returns the name of the most predictive feature for the model

    Args:
         importance_df: A DataFrame of features and their raw and absolute value coefficients in a model

    Returns:
        the feature name with the largest absolute value coefficient
    """
    best_feature = importance_df.loc[importance_df["Abs_Coefficient"].idxmax()]
    return best_feature["Feature"]

def tally_top_features(models, events):
    """
    Counts how many events each feature is the single most predictive for, defined as
    the feature with the largest absolute value coefficient

    Args:
        models: dict of events to model info
        events: the events to tally across

    Return:
        a dictionary mapping feature names to number of events it was a top feature for
    """
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
    """
    Prints training statistics (MAE, MSE, R^2, top feature), for a single event's model

    Args:
        event: the event name
        model: a single event's model information dict
    """

    stats = model["stats"]
    best_feature = get_top_feature(model["importance"])
    print(f"Printing training statistics for {event}")
    print(f"    MAE: {model["stats"]["mae"]:.3f} seconds")
    print(f"    MSE: {model["stats"]["mse"]:.3f} seconds")
    print(f"    R^2: {model["stats"]["r2"]:.3f}")
    print(f"    Most predictive feature: {best_feature}")


def print_feature_ranking(ranked, total):
    """
    Prints the ranked list of the most-predictive features across all given models

    Args:
        ranked: list of (feature, count) tuples, sorted my most to least predictive
        total: total number of models the ranking is based on
    """

    print("\n")
    max_feature, max_count = ranked[0]
    print(f"Most predictive feature for all models: {max_feature} ({max_count}/{total})")

    for i, (feature, count) in enumerate(ranked[1:], start=2):
        print(f"    #{i} most predictive feature for models: {feature} ({count}/{total})")


def feature_pie_plot(ranked, total, title):
    """
    Creates and saves a pie chart of the feature predictiveness count to a file

    Args:
        ranked: list of (feature, count) tuples, sorted by most to least predictive
        total: the number of models the ranking is based on
        title: the label used in the chart title and output file name
    """

    all_features = [item[0] for item in ranked]
    all_counts = [item[1] for item in ranked]
    plt.pie(all_counts, labels=all_features, autopct='%1.1f%%')
    plt.title(f'Most predictive features for {title} (n={total})')
    # plt.legend(title="Predictive Features")
    plt.savefig(f"figures/feature_pie_plot_{title.replace(" ", "_")}.png", dpi=300, bbox_inches="tight")
    plt.close()

def get_correlogram(models):
    """
    Saves a correlogram (feature correlation heatmap) for every event's model

    Args:
        models: a dictionary of events to model info dicts
    """

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
    """
    Prints per-event model status plus overall most-predictive feature ranking

    Args:
        models: a dict of events to model info dicts
        events: the events included in the report
        title: the label for the report
    """
    print("\n")
    for event in events:
        print_model_stats(event, models[event])

    #tally which features are the top features, rank, then print
    features = tally_top_features(models, events)
    ranked = rank_features(features)
    print_feature_ranking(ranked, len(events))

    #show all graphics
    # feature_pie_plot(ranked, len(events), title)
    # get_correlogram(models)
