# ⚽ World Cup Match Simulator

A web app that predicts scores between any two national football teams from any era, using a Random Forest machine learning model trained on international match history.

Example: **Brazil 2002 vs Argentina 2022** → get a predicted scoreline, win probability breakdown, and head-to-head history between the two teams.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-web%20framework-black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)

---

## Features

- 🔮 **Score prediction** — Random Forest regression model predicts the final score of any matchup
- 📊 **Win probability** — derived from how the 100 individual decision trees in the forest vote (home win / draw / away win)
- 📈 **Team stats** — attack rating (avg goals scored) and defense rating (avg goals conceded), calculated per team per year
- 🤝 **Head-to-head history** — full match history between any two teams, with a win/draw/loss summary
- 🏆 **World Cup team filter** — enter a year and only see teams that actually played in that World Cup
- 🎨 **Dark FIFA-style UI** — black background, gold accents, live country flags via [flagcdn.com](https://flagcdn.com)

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn (`RandomForestRegressor`) |
| Data handling | pandas, numpy |
| Model persistence | joblib |
| Frontend | HTML, CSS, vanilla JavaScript |
| Flags | flagcdn.com API |
| Data | [International football results, 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle) |

---

## How It Works

### The ML model

1. Loads international match results (1872–present) from `results.csv`
2. For each team, in each year, calculates:
   - **Attack rating** — average goals scored in all matches up to that year
   - **Defense rating** — average goals conceded in all matches up to that year
3. Caches these stats to avoid recalculating them on every request
4. Trains two separate `RandomForestRegressor` models (100 trees each):
   - One predicts the home team's goals
   - One predicts the away team's goals
5. Features used per match: home attack, home defense, away attack, away defense, is World Cup (1/0), is neutral venue (1/0)
6. Model is saved to disk with `joblib` — loads instantly on future runs instead of retraining

### Win probability

Rather than just averaging the forest's prediction into a single score, the API also looks at how each of the 100 individual trees "voted" — if a tree predicts more home goals than away goals, that's a vote for a home win, and so on. Tallying those votes across all 100 trees gives a genuine win / draw / loss probability split.

### Head-to-head history

A separate endpoint filters the full match dataset for every past meeting between the two selected teams (regardless of year), and returns a win/draw/loss tally plus the 10 most recent matches.

---

## Project Structure

```
World-Cup-simulator/
├── app.py                 # Flask server & API routes
├── predictor.py            # ML model: training, caching, prediction logic
├── templates/
│   └── index.html          # Frontend UI
├── results.csv              # Match data (NOT included — see setup below)
├── model_home.pkl           # Trained home-goals model (generated, not included)
├── model_away.pkl           # Trained away-goals model (generated, not included)
├── cache.pkl                 # Cached team stats (generated, not included)
└── README.md
```

> `results.csv` and the `.pkl` model files are excluded from this repo via `.gitignore` since they're either too large or easily regenerated. See setup instructions below.

---

## API Reference

### `GET /api/predict`
Predicts a match score between two teams.

**Query params:** `team1`, `year1`, `team2`, `year2`

**Example:** `/api/predict?team1=Brazil&year1=2002&team2=Argentina&year2=2022`

```json
{
  "team1": "Brazil",
  "year1": 2002,
  "team2": "Argentina",
  "year2": 2022,
  "score1": 2,
  "score2": 1,
  "stats1": { "attack": 2.18, "defense": 0.90, "total_matches": 1064 },
  "stats2": { "attack": 2.05, "defense": 0.95, "total_matches": 1020 },
  "win_prob": { "team1": 57.0, "draw": 22.0, "team2": 21.0 }
}
```

### `GET /api/teams`
Returns all teams in the dataset (328 teams).

### `GET /api/wc_teams`
Returns only the teams that played in a specific World Cup year.

**Query params:** `year`

**Example:** `/api/wc_teams?year=2022`

### `GET /api/head_to_head`
Returns full match history and a win/draw/loss tally between two teams.

**Query params:** `team1`, `team2`

**Example:** `/api/head_to_head?team1=England&team2=Brazil`

---

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/anand355/World-Cup-simulator.git
cd World-Cup-simulator
```

**2. Install dependencies**
```bash
pip install flask pandas scikit-learn numpy joblib
```

**3. Download the dataset**

This project uses the [International football results dataset](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) from Kaggle (requires a free Kaggle account).

Download it, unzip it, and place `results.csv` directly in the project folder (same level as `app.py`).

**4. Train the model**
```bash
python predictor.py
```
This builds and saves `model_home.pkl`, `model_away.pkl`, and `cache.pkl`. It only needs to be run once — future runs of `app.py` will load the saved model instantly. Re-run this any time `results.csv` is updated with newer data.

**5. Start the server**
```bash
python app.py
```

**6. Open the app**

Visit **http://localhost:5000** in your browser.

---

## Roadmap

- [x] ML score prediction
- [x] World Cup team filtering by year
- [x] Country flags
- [x] Dark FIFA-style UI
- [x] Result card with stats
- [x] Win probability display
- [x] Head-to-head history
- [ ] Deploy to Render

---

## Data Source

Match data from [martj42/international-football-results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) on Kaggle — an actively maintained dataset of international football results dating back to 1872.

---

## License

This project is for educational purposes as part of a personal learning project in Python, Flask, and machine learning.
