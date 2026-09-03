from pathlib import Path
from playwright.sync_api import sync_playwright
import json
import sys
import time
import random
import urllib


BATCH_START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BATCH_END   = int(sys.argv[2]) if len(sys.argv) > 2 else 100

ALL_ATHLETES = {r["memberId"]: r["fullName"] for r in json.loads(Path(
    "../data_jsons/athlete_ids.json").read_text(encoding="utf-8"))}
print(len(ALL_ATHLETES))
ATHLETES = dict(list(ALL_ATHLETES.items())[BATCH_START:BATCH_END])  # only this slice this run

REGIONS = {r["lscCode"]: r["lscName"] for r in json.loads(Path(
    "../data_jsons/regions.json").read_text(encoding="utf-8"))}

TIME_API = "GetAllTimesForFilters"
MEMBER_APIS =  {"memberProfile" : "GetMember", "memberCourses": "GetSwimmerCourses", "memberSeasons": "GetSwimmerSeasons", "memberEvents": "GetSwimmerEvents"}


def load_swimmer_profile(page, member_url):


    #general profile, recorded seasons, events swam
    athlete_profile = {}

    for api_key in MEMBER_APIS:
        api = MEMBER_APIS[api_key]
        athlete_profile[api_key] = do_request(page, member_url, api)

    if not athlete_profile.get('memberProfile') or not athlete_profile.get('memberEvents') or not athlete_profile.get(
            'memberCourses'):
        print(f"Skipping {member_url} — missing core profile data (likely repeated timeouts)")
        return None

    #creates the correct payload needed to parse the search url
    parsed_events = build_swimmer_events(athlete_profile['memberEvents'])

    #print(f'Scraping profile for {athlete_profile["memberProfile"]["shortName"]}')

    times = scrape_swimmer_times(page, member_url, athlete_profile["memberCourses"], parsed_events)

    parsed_profile = parse_profile(athlete_profile)
    parsed_profile['times'] = times

    #print(parsed_profile)
    return parsed_profile

def parse_profile(athlete_profile):

    parsed_profile = {}

    #Get Member Formatting:
    for key in athlete_profile['memberProfile']:

        if key != 'profilePicUrl':
            if key== 'lscCode':
                parsed_profile['region'] = REGIONS.get(athlete_profile['memberProfile'][key],
                                                       athlete_profile['memberProfile'][key])

            if key == 'isNcaa':
                parsed_profile[key] = bool(athlete_profile['memberProfile'][key])

            else:
                parsed_profile[key] = athlete_profile['memberProfile'][key]

    #Get all courses they did
    courses = [course["courseCode"] for course in athlete_profile["memberCourses"]]
    parsed_profile['courses'] = courses

    #Get seasons they have swam
    seasons = athlete_profile['memberSeasons']
    parsed_profile['seasons'] = {"numSeasons": len(seasons), "seasonsList": seasons}

    return parsed_profile


def scrape_swimmer_times(page, url, courses, events):

    time_dict = {}

    #iterate through each course
    for entry in courses:
        courseCode = entry["courseCode"]
        #print(f'    Scraping all {courseCode} events...')

        #iterate for every event a swimmer has
        for event in events:
            event_id = events[event]
            scrape_url = build_times_url(url, courseCode, event_id)
            time_data = do_request(page, scrape_url, TIME_API)

            if time_data is None:
                continue

            #get the eventCode
            eventCode = time_data[0]['eventCode']
            parsed_time_data = []

            excluded_fields = ['swimTimeId', 'meetId', 'powerPoints', 'eventCode', 'courseCode']

            for time_entry in time_data:
                parsed_time_entry = {}

                for field in time_entry:
                    if field not in excluded_fields:
                        parsed_time_entry[field] = time_entry[field]

                parsed_time_data.append(parsed_time_entry)

            time_dict[eventCode] = {"numSwims": len(parsed_time_data) ,"times": parsed_time_data}
            time.sleep(random.uniform(0.3, 0.7))

    return time_dict



#####################################################
#################### HELPER CODE ####################
#####################################################

##### URL PARSING CODE #####
def get_swimmer_url(memberId):
    '''

    :param memberId: a swimmer's member ID
    :return: a URL for the swimmer's USA swimming profile, showing all times for every event swam
    '''
    return f"https://data.usaswimming.org/search/athlete/{memberId}/all-times?&sortBy=Newest"


def build_times_url(member_url, course, event_ids):
    '''

    :param member_url: a URL for the swimmer's USA swimming profile, showing all times for every event swam
    :param course: the event course - SCY, SCM, or LCM
    :param event_ids: the URL formatted event ids
    :return: a search URL for the swimmers times for a given event
    '''

    before, sep, after = member_url.partition("all-times?")
    first_half = before + sep
    second_half = after
    encoded_events = urllib.parse.quote(event_ids)

    return f'{first_half}course={course}&events={encoded_events}{second_half}'


def build_swimmer_events(events):

    swimmer_parsed_events = {}

    if not events:
        return swimmer_parsed_events

    for entry in events:
        stroke_name = entry["strokeName"]
        distance = entry["distance"]
        event_id = entry["eventId"]

        parsed_event_name = f'{stroke_name}_{distance}'

        if parsed_event_name not in swimmer_parsed_events:
            swimmer_parsed_events[parsed_event_name] = f'{parsed_event_name}_{event_id}'
        else:
            swimmer_parsed_events[parsed_event_name] += f',{event_id}'


    return swimmer_parsed_events


###### NETWORK REQUEST HELPER #####

def do_request(page, url, api, retries=3):
    for attempt in range(1, retries + 1):
        try:
            with page.expect_response(
                    lambda r: api in r.url, timeout=60000
            ) as response_info:
                page.goto(url)

            response = response_info.value
            if response.status != 200:
                if response.status == 404:
                    return None
                else:
                    print(f"Status {response.status} for {url}")
                    return None
            data = response.json()
            return data

        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed on {url}: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)  # backoff: 3s, 6s, 9s...
            else:
                with open("../athlete_times_jsons/failed_urls.txt", "a") as f:
                    f.write(url + "\n")
                return None
    return None


def main():
    overall_dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        counter = 1

        for memberId in ATHLETES:
            member_url = get_swimmer_url(memberId)

            try:
                profile = load_swimmer_profile(page, member_url)
                if profile is not None:
                    overall_dict[memberId] = profile
            except Exception as e:
                print(f"FAILED athlete {memberId}: {type(e).__name__}: {e}")

            print(f'Finished scraping athlete {counter}/{len(ATHLETES)}')
            counter+=1

            with open(f"athlete_times_{BATCH_START}_{BATCH_END}_partial.json", "w", encoding="utf-8") as f:
                json.dump(list(overall_dict.values()), f, ensure_ascii=False, indent=2)

            time.sleep(random.uniform(1.0, 2.0))

        #load_swimmer_profile(page,"https://data.usaswimming.org/search/athlete/CF5A195D755242/all-times?&sortBy=Newest")
        browser.close()

    overall_list = sorted(overall_dict.values(), key=lambda a: a.get("fullName") or "")

    with open(f"athlete_times_{BATCH_START}_{BATCH_END}.json", "w", encoding="utf-8") as f:
        json.dump(overall_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
