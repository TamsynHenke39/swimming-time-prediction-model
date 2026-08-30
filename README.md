# Swimming Time Prediction Model
 
Scrapes historical swim time data from **USA Swimming** and trains machine learning models (via scikit-learn) to predict a swimmer's next swim time based on their performance history. Requires manually logging into an account on the site before scraping.
 
## Requirements
 
- Python >= 3.13
## Dependencies
 
- [matplotlib](https://matplotlib.org/) >= 3.11.1
- [numpy](https://numpy.org/) >= 2.5.1
- [pandas](https://pandas.pydata.org/) >= 3.0.5
- [playwright](https://playwright.dev/python/) >= 1.61.0
- [scikit-learn](https://scikit-learn.org/) >= 1.9.0
## Installation
 
Using [uv](https://docs.astral.sh/uv/) (recommended, since this project uses a `pyproject.toml`):
 
```bash
uv sync
```
 
Or with pip:
 
```bash
pip install -e .
```
 
Then install Playwright's browser binaries:
 
```bash
playwright install
```
 
## Authentication
 
This project requires an account on **USA Swimming**. Scraping is done via Playwright, which opens a browser session — you'll need to manually log in each time you run one of the scraping scripts, since sessions/cookies are not persisted between runs.
 
1. Create an account on USA Swimming if you don't already have one.
2. Run the desired scraping script (e.g. `python athlete_scraper.py`).
3. A browser window will open — log in manually when prompted.
4. Once logged in, the scraper will proceed to collect data.
> **Note:** You'll need to repeat the manual login step every time you run a scraping script.
 
## Data Collection Pipeline
 
Scraping happens in two stages and must be run in order:
 
1. **Collect athlete IDs** — run `athlete_scraper` first to build a list of athlete names and their corresponding IDs on USA Swimming, sampled from the top 100 swimmers in each event (no duplicates) across all distances, courses, strokes, and gender.
```bash
   python athlete_scraper.py
```
 
2. **Collect swim times** — using the athlete IDs from step 1, run `capture_swim_api` to pull every event and time for each athlete.
```bash
   python capture_swim_api.py
```
 
> **Note:** Scraping is slow, since each athlete's data is collected via a logged-in browser session (see [Authentication](#authentication)). It's recommended to run `capture_swim_api` in batches of **~50 athletes** at a time rather than the full list at once. This will take ~1.5-2 hours for each batch.
 
## Usage
 
Once data has been collected, run the prediction script to train the multi-linear regression models for each event:
 
```bash
python mlr.py
```
  
## License
 
_TBD._
