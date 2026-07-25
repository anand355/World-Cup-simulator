from flask import Flask, jsonify, request, render_template
from predictor import load_data, build_model, load_model, save_model, predict_score
import os

app = Flask(__name__)

# Load data and model when server starts
print("Loading data...")
df = load_data()
model_home, model_away, cache = load_model()

if model_home is None:
    model_home, model_away, cache = build_model(df)
    save_model(model_home, model_away, cache)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict")
def predict():
    team1 = request.args.get("team1")
    year1 = int(request.args.get("year1"))
    team2 = request.args.get("team2")
    year2 = int(request.args.get("year2"))
    
    # Get team stats from cache
    stats1 = cache.get(f"{team1}_{year1}")
    stats2 = cache.get(f"{team2}_{year2}")
    
    if not stats1 or not stats2:
        return jsonify({"error": "Team or year not found"})
    
    # Predict score
    import numpy as np
    features = np.array([[
        stats1['attack'],
        stats1['defense'],
        stats2['attack'],
        stats2['defense'],
        1, 1
    ]])
    
    home_preds = [tree.predict(features)[0] for tree in model_home.estimators_]
    away_preds = [tree.predict(features)[0] for tree in model_away.estimators_]

    home_goals = round(sum(home_preds) / len(home_preds))
    away_goals = round(sum(away_preds) / len(away_preds))

    home_wins = sum(1 for h, a in zip(home_preds, away_preds) if h > a)
    away_wins = sum(1 for h, a in zip(home_preds, away_preds) if h < a)
    draws = len(home_preds) - home_wins - away_wins
    n = len(home_preds)

    return jsonify({
        "team1": team1,
        "year1": year1,
        "team2": team2,
        "year2": year2,
        "score1": int(home_goals),
        "score2": int(away_goals),
        "stats1": {
            "attack": float(stats1['attack']),
            "defense": float(stats1['defense']),
            "total_matches": int(stats1['total_matches'])
        },
        "stats2": {
            "attack": float(stats2['attack']),
            "defense": float(stats2['defense']),
            "total_matches": int(stats2['total_matches'])
        },
        "win_prob": {
            "team1": round(home_wins / n * 100, 1),
            "draw": round(draws / n * 100, 1),
            "team2": round(away_wins / n * 100, 1)
        }
    })
@app.route("/api/teams")
def get_teams():
    teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    return jsonify(teams)
@app.route("/api/wc_teams")
def get_wc_teams():
    year = int(request.args.get("year"))
    
    wc = df[(df['tournament'] == 'FIFA World Cup') & (df['year'] == year)]
    
    home_teams = set(wc['home_team'].unique())
    away_teams = set(wc['away_team'].unique())
    
    all_teams = sorted(home_teams | away_teams)
    
    return jsonify(all_teams)

@app.route("/api/head_to_head")
def head_to_head():
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")

    matches = df[
        ((df['home_team'] == team1) & (df['away_team'] == team2)) |
        ((df['home_team'] == team2) & (df['away_team'] == team1))
    ].dropna(subset=['home_score', 'away_score']).sort_values('date', ascending=False)

    team1_wins = 0
    team2_wins = 0
    draws = 0
    match_list = []

    for _, row in matches.iterrows():
        home = row['home_team']
        away = row['away_team']
        hs = int(row['home_score'])
        as_ = int(row['away_score'])

        if hs == as_:
            draws += 1
        elif (home == team1 and hs > as_) or (away == team1 and as_ > hs):
            team1_wins += 1
        else:
            team2_wins += 1

        match_list.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "tournament": row['tournament']
        })

    return jsonify({
        "team1": team1,
        "team2": team2,
        "total_matches": len(match_list),
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "draws": draws,
        "matches": match_list[:10]
    })
if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)