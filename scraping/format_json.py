import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

INPUT_DIR = Path("../athlete_times_jsons/complete_scraped_times_jsons")
OUTPUT_CSV = "training_data.csv"
PROCESSED_LOG = "processed_files.txt"

###### General calculation helper methods ######
def parse_time(time):
    """
    Convert swimming time to total seconds.
    Removes any trailing letters like 'r'.

    Args:
        time: a race time, formatted as MM:SS.MS

    Returns:
        swim time in total seconds
    """

    if time is None:
        return None

    time = time.strip()

    # Remove trailing non-numeric characters
    while time and not (time[-1].isdigit() or time[-1] == "."):
        time = time[:-1]

    if ":" in time:
        minutes, seconds = time.split(":")
        return int(minutes) * 60 + float(seconds)

    return float(time)

def get_std_previous(times):
    """
    Gets the standard deviation of the previous 3 swims.
    If fewer than 2, return None

    Args:
        times: list of swim times (already parsed to seconds)

    Returns:
        standard deviation of the previous 3 times, None if fewer than 2 times
    """

    recent = times[-3:]

    if len(recent) < 2:
        return None

    return float(np.std(recent, ddof=0))


def calculate_years_competing(history):
    """

    Calculates the number of years competing from a given race history

    Args:
        history: a swimmer's race history in an event, ordered from oldest to newest

    Returns:
        the number of years between the first and most recent swim in history, 0 if history is empty
    """

    if len(history) == 0:
        return 0

    first = datetime.fromisoformat(history[0]["swimDate"])
    latest = datetime.fromisoformat(history[-1]["swimDate"])

    return (latest-first).days / 365.25


###### Feature and row building helpers ######

def compute_prediction_features(curr_swim, history):
    """

    Args:
        curr_swim: the swim data point being featurized
        history: all swims before curr_swim, oldest to newest

    Returns:
        a dictionary of prediction features for the given swim
            previousTime: the last time swam
            averageLast3: average of last 3 times in the event, -
            bestPrevious: best time in the event history
            stdPrevious - standard deviation of previous 3 swims
            daysSinceBest - number of days since the last race
            daysSinceLast - number of days since the last swim in the event
            ageAtBest - age of swimmer (whole years) on the date of the best time
    """
    previous_times = [parse_time(swim["swimTime"]) for swim in history]

    current_date = datetime.fromisoformat(curr_swim["swimDate"])

    # calculate previous performance features features

    if history:

        # most recent swim before this one
        previous_time = previous_times[-1]

        # averge of previous 3 swims
        average_last3 = np.mean(previous_times[-3:])

        # days since previous swim
        days_since_last = (current_date - datetime.fromisoformat(history[-1]["swimDate"])).days

        # gives us the best swim OBJECT
        best_swim = min(history, key=lambda x: parse_time(x["swimTime"]))

        # best time overall
        best_previous = parse_time(best_swim["swimTime"])

        # days since they swam there best swim
        days_since_best = (current_date - datetime.fromisoformat(best_swim["swimDate"])).days

        # age when they swam their best swim
        age_at_best = (best_swim["swimmerAge"])

    else:
        previous_time = None
        average_last3 = None
        best_previous = None
        days_since_last = None
        days_since_best = None
        age_at_best = None

    return {
        "previousTime": previous_time,
        "averageLast3": average_last3,
        "bestPrevious": best_previous,
        "stdPrevious": get_std_previous(previous_times),
        "daysSinceBest": days_since_best,
        "daysSinceLast": days_since_last,
        "ageAtBest": age_at_best,
    }

def build_training_row(athlete_info, event_name, curr_swim, history):
    """
    Builds a single row (one per swim) for a given swimmer/event/swim

    Args:
        athlete_info: a dict of athlete profile information (memberId, fullName, region, isNcaa, currentAge)
        event_name:  the name of the event, ex: "100 FR SCY"
        curr_swim: the swim the row represents
        history: all of the swimmer's swims in the event before curr_swim

    Returns:
        a dict representing one CSV row of the data
    """

    distance, stroke, course = event_name.split()

    row = {
        # Current profile
        "memberId": athlete_info["memberId"],
        "fullName": athlete_info["fullName"],

        # Current profile
        "region": athlete_info["region"],
        "isNcaa": athlete_info["isNcaa"],
        "currentAge": athlete_info["currentAge"],

        # swim information
        "event": event_name,
        "distance": distance,
        "stroke": stroke,
        "course": course,

        # swim metadata
        "meetName": curr_swim["meetName"],
        "swimDate": curr_swim["swimDate"],
        "swimTime": parse_time(curr_swim["swimTime"]),

        "ageAtSwim": curr_swim["swimmerAge"],
        "numPreviousSwims": len(history),
        **compute_prediction_features(curr_swim, history),
        "yearsCompeting": calculate_years_competing(history),
    }

    return row


###### Athlete & file processing helper methods ######

def process_event(athlete_info, event_name, event_data):
    """
    Builds one training row per swim for a single swimmer/event

    Args:
        athlete_info: a dict of current profile information for a swimmer
        event_name: the name of the event, ex: "100 FR SCY"
        event_data: an event's raw, scraped data, containing a list of times ordered newest to oldest

    Returns:
        a list of training rows, one per swim in the event
    """

    print(f"    Event: {event_name}")

    # reverse, in order oldest --> newest
    swims = list(reversed(event_data["times"]))
    rows = []

    # one training row per swim
    for i, curr_swim in enumerate(swims):
        print(f"        Swim {i + 1}/{len(swims)}")

        # everything before this swim, the entire swim object
        history = swims[:i]
        rows.append(build_training_row(athlete_info, event_name, curr_swim, history))

    return rows

def process_athlete(athlete):
    """
    Builds all training rows for a single athlete, across every event they've swam

    Args:
        athlete: a single athlete's scraped profile + dictionary of race times

    Returns:
        a list of training row dictionaries for the athlete
    """

    print( f"Athlete: {athlete['fullName']}")
    athlete_info =  {
        "memberId" : athlete["memberId"],
        "fullName" : athlete["fullName"],
        "region" : athlete["region"], #TODO: one hot encoding for the regions
        "isNcaa" : athlete["isNcaa"],
        "currentAge" : athlete["swimmerAge"],
    }

    rows = []

    for event_name, event_data in athlete["times"].items():
        rows.extend(process_event(athlete_info, event_name, event_data))


    return rows

def process_file(json_file):
    """
    Builds training rows contained in a single JSON file of scraped-athlete race/time data

    Args:
        json_file: path to JSON file containing a batch of athlete profiles & times

    Returns:
        a list of training row dictionaries for every athlete in the file
    """

    print(f"Processing {json_file.name}")

    with open(json_file, encoding="utf-8") as f:
        athletes = json.load(f)

    rows = []

    for athlete in athletes:
        rows.extend(process_athlete(athlete))

    return rows

###### File-tracking helper methods ######

def load_processed_files(log_path):

    """
    Loads the set of already-processed JSON files from the processed-files log

    Args:
        log_path: the path to the processed files log

    Returns:
        a set of filenames that have already been processed; empty set if no log; file names should be of form
        "athlete_times_{BATCH_START}_{BATCH_END}"
    """

    if not os.path.exists(PROCESSED_LOG):
        return set()

    with open(PROCESSED_LOG, encoding="utf-8") as f:
        return {line.strip() for line in f }

def mark_files_as_processed(log_path, filenames):
    """
    Adds newly processed filenames to the processed files log

    Args:
        log_path: the path to the processed files log
        filenames: the filenames to record as processed; each should be named "athlete_times_{BATCH_START}_{BATCH_END}"
    """

    with open(log_path,"a") as f:
        for filename in filenames:
            f.write(filename + "\n")

def save_rows_to_csv(rows, output_csv):
    """
    Writes training rows to the output CSV, appending if it already exists

    Args:
        rows: the list of training row dictionaries
        output_csv: the path the output CSV file

    Returns:
        the DataFrame that was written
    """

    df = pd.DataFrame(rows)
    if os.path.exists(OUTPUT_CSV):

        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    else:
        df.to_csv(OUTPUT_CSV, index=False)

    return df

def main():
    processed_files = load_processed_files(PROCESSED_LOG)
    rows = []
    new_processed = []

    #process all the files
    for json_file in sorted(INPUT_DIR.glob("*.json")):

        if json_file.name in processed_files:
            continue
        rows.extend(process_file(json_file))
        new_processed.append(json_file.name)

    if rows:
        df = save_rows_to_csv(rows, OUTPUT_CSV)
        mark_files_as_processed(PROCESSED_LOG, new_processed)
        print(f"Added {len(df)} rows")
    else:
        print("No new files found.")

if __name__=="__main__":
    main()