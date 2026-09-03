import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

INPUT_DIR = Path("../athlete_times_jsons/complete_scraped_times_jsons")
OUTPUT_CSV = "training_data.csv"
PROCESSED_LOG = "processed_files.txt"


# #do one-hot encoding on the categorical data for the regions
#
# region_data = [r["lscName"] for r in json.loads(Path("regions.json").read_text(encoding="utf-8"))]
# region_df = pd.DataFrame(region_data)
# encoded_region_df = pd.get_dummies(region_df, drop_firt=True)



def parse_time(time):
    """
    Convert swimming time to total seconds.
    Removes any trailing letters like 'r'.
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
    If fewer than 2, return 0
    """

    recent = previous_times[-3:]

    if len(recent) < 2:
        return None

    return float(np.std(recent, ddof=0))

# def improvement(history):
#     """
#     Shows improvement in seconds/year
#     Negative = improving
#     """
#
#     if len(history) < 2:
#         return 0.0
#
#     first_date = datetime.fromisoformat(history[0]["swimDate"])
#
#     years = []
#     times = []
#
#     for swim in history:
#         date = datetime.fromisoformat(swim["swimDate"])
#
#         years_since_first = ( date - first_date).days / 365.25
#
#         years.append(years)
#         times.append(parse_time(swim['swimTime']))
#
#     print("Entering polyfit")
#     print(years)
#     print(times)
#
#     slope = np.polyfit(years, times, 1)[0]
#
#     print("Leaving polyfit")
#
#     return float(slope)

def calculate_years_competing(history):

    if len(history) == 0:
        return 0

    first = datetime.fromisoformat(history[0]["swimDate"])
    latest = datetime.fromisoformat(history[-1]["swimDate"])

    return (latest-first).days / 365.25

processed_files = set()

if os.path.exists(PROCESSED_LOG):
    with open(PROCESSED_LOG, encoding="utf-8") as f:
        processed_files = {
            line.strip()
            for line in f
        }


rows = []
new_processed = []

for json_file in sorted(INPUT_DIR.glob("*.json")):

    if json_file.name in processed_files:
        continue

    print(f"Processing {json_file.name}")

    with open(json_file, encoding="utf-8") as f:
        athletes = json.load(f)

    for athlete in athletes:

        print(
            f"Athlete: {athlete['fullName']}"
        )

        #current profile information, known today

        member_id = athlete["memberId"]
        full_name = athlete["fullName"]
        region = athlete["region"] #TODO: one hot encoding for the regions
        is_ncaa = athlete["isNcaa"]
        current_age = athlete["swimmerAge"]

        #go through all an athlete's events
        for event_name, event_data in athlete["times"].items():

            print(
                f"    Event: {event_name}"
            )

            #reverse, in order oldest --> newest
            swims = list(reversed(event_data["times"]))

            #one training row per swim
            for i, curr_swim in enumerate(swims):

                print(
                    f"        Swim {i + 1}/{len(swims)}"
                )

                #everything before this swim, the entire swim object
                history = swims[:i]

                #convert previous swims into numbers
                previous_times = [
                    parse_time(swim["swimTime"]) for swim in history
                ]

                current_date = datetime.fromisoformat(curr_swim["swimDate"])

                #calculate previous performance features features

                if history:

                    #most recent swim before this one
                    previous_time = previous_times[-1]

                    #averge of previous 3 swims
                    average_last3 = np.mean(previous_times[-3:])

                    #days since previous swim
                    days_since_last = (current_date - datetime.fromisoformat(history[-1]["swimDate"])).days

                    #gives us the best swim OBJECT
                    best_swim = min(history, key=lambda x: parse_time(x["swimTime"]))

                    #best time overall
                    best_previous = parse_time(best_swim["swimTime"])

                    #days since they swam there best swim
                    days_since_best = (current_date - datetime.fromisoformat(best_swim["swimDate"])).days

                    #age when they swam their best swim
                    age_at_best = (best_swim["swimmerAge"])

                else:
                    previous_time = None
                    average_last3 = None
                    best_previous = None
                    days_since_last = None
                    days_since_best = None
                    age_at_best = None

                parts = event_name.split()
                distance = parts[0]
                stroke = parts[1]
                course = parts[2]

                #add a CSV row

                rows.append({
                    "memberId": member_id,
                    "fullName": full_name,

                    # Current profile

                    "region": region,
                    "isNcaa": is_ncaa,
                    "currentAge": current_age,

                    #swim information
                    "event": event_name,
                    "distance": distance,
                    "stroke": stroke,
                    "course": course,

                    #swim metadata
                    "meetName": curr_swim["meetName"],
                    "swimDate": curr_swim["swimDate"],
                    "swimTime": parse_time(curr_swim["swimTime"]),

                    #prediction features
                    "ageAtSwim": curr_swim["swimmerAge"],
                    "numPreviousSwims": len(history),
                    "previousTime": previous_time,
                    "averageLast3": average_last3,
                    "bestPrevious": best_previous,
                    "stdPrevious": get_std_previous(previous_times),
                    "daysSinceBest": days_since_best,
                    "daysSinceLast": days_since_last,
                    "ageAtBest": age_at_best,
                    "yearsCompeting": calculate_years_competing(history)

                })
    new_processed.append(json_file.name)


if rows:


    df = pd.DataFrame(rows)


    if os.path.exists(OUTPUT_CSV):

        df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=False,
            index=False
        )


    else:

        df.to_csv(
            OUTPUT_CSV,
            index=False
        )



    with open(
        PROCESSED_LOG,
        "a"
    ) as f:

        for file in new_processed:

            f.write(
                file + "\n"
            )



    print(
        f"Added {len(df)} rows"
    )


else:

    print(
        "No new files found."
    )

