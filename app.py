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
    
    home_goals = round(model_home.predict(features)[0])
    away_goals = round(model_away.predict(features)[0])
    
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

if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True)