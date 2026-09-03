from pathlib import Path

from playwright.sync_api import sync_playwright
import json
import time

# Open the file and load the JSON array
EVENT_LIST = json.loads(Path("../data_jsons/all_events.json").read_text())
TIME_API = "GetTopTimesLeaderBoard"


def get_top_times_url(eventId, course, genderID):
    """Generate a formatted URL for searching the USA Swimming database.

    Retrieves top times for a specific swimming event based on the course,
    gender, and event ID.

    Args:
        eventID (str): unique numerical event identifier for each swim event
        course (str): Type of pool used (e.g., 'SCY', 'SCM', 'LCM').
        genderID (int): ID for the gender, 1 for male, 2 for female

    Returns:
        str: formatted URL pointing to USA swimming database
    """
    return f'https://data.usaswimming.org/leaderboards/top-times?eventId={eventId}&course={course}&competitionGenderTypeId={genderID}'


def scrape_athlete_ids(page):
    """Scrapes athelete ids for all events and gender.

    Generates URLs for men's and women's leaderboards for all events, accumulates
    dictionary of unique athelete profiles

    Args:
        page (Playwright.sync_api.Page): The Playwright browswer tab or window

    Returns:
        dict: A dictionary of athletes, aggregated by member ID
    """

    athletes_dict = {}

    for event in EVENT_LIST:
        eventId = event.get("eventId")
        courseCode = event.get("courseCode")
        print("Scraping times for " + event.get("eventCode") + "...")

        mens_url = get_top_times_url(eventId, courseCode, 1)
        womens_url = get_top_times_url(eventId, courseCode, 2)

        old_size = len(athletes_dict)


        scrape_leaderboard(page, mens_url, athletes_dict)
        print(f"  Finished scraping men's {event.get("eventCode")}")
        scrape_leaderboard(page, womens_url, athletes_dict)
        print(f"  Finished scraping women's {event.get("eventCode")}")
        time.sleep(1)

        print(f"Added {len(athletes_dict) - old_size} to dictionary \n")


    return athletes_dict


def scrape_leaderboard(page, scrape_url,  athletes_dict):
    """Scrapes the JSON request data from the given URL.

    Captures the top times network response from an event leaderboard (by gender)
    and adds each unique athlete ID and name to the provided dictionary

    Args:
        page (playwright.sync_api.Page): The Playwright browser tab or window.
        scrape_url (str): The URL from which athlete data will be fetched.
        athletes_dict (dict): A dictionary storing athlete IDs and names.
    """
    try:
        with page.expect_response(
                lambda r: TIME_API in r.url, timeout=45000
        ) as response_info:
            page.goto(scrape_url)

        response = response_info.value
        if response.status != 200:
            print(f"Status {response.status} for {scrape_url}")
            return
        data = response.json()

        for entry in data:
            memberId = entry.get("memberId")

            if memberId not in athletes_dict:
                athletes_dict[memberId] = {
                    "memberId": memberId,
                    "fullName": entry.get("fullName"),
                }

    except Exception as e:
        print(f"Failed on {scrape_url}: {type(e).__name__}: {e}")


def main():
    """
    Initiates the scraping process using Playwright.

    Launches a Chromimum browswer instance, gathers all athlete data,
    sorts the unique ahtletes by name, and outputs the results to 'athlete_ids.json'
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        athletes_dict = scrape_athlete_ids(page)
        browser.close()

    athlete_list = sorted(athletes_dict.values(), key=lambda a: a.get("fullName") or "")

    with open("../data_jsons/athlete_ids.json", "w", encoding="utf-8") as f:
        json.dump(athlete_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
