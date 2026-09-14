import mlr
import model_reporting
from event_filter_client import get_custom_data


def main():
    """
    Trains models for all events, lets user filter to subset of models and print report
    of subset stats and top features

    Results are displayed by R^2, from best to worst-fitting model
    """
    df = mlr.load_training_data()
    events, distances, strokes, courses = mlr.get_events(df)

    models = mlr.build_all_models(df, events)
    sorted_events = model_reporting.sort_events_by_r2(models)

    model_subset, events_subset, title = get_custom_data(models, sorted_events, distances, strokes, courses)

    model_reporting.print_report(model_subset, events_subset, title)

if __name__ == "__main__":
    main()