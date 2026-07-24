import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os

def load_data():
    df = pd.read_csv('results.csv')
    
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Add a year column
    df['year'] = df['date'].dt.year
    
    print(f"Loaded {len(df)} matches")
    print(f"From {df['year'].min()} to {df['year'].max()}")
    
    return df

def calculate_team_stats(df, team, year):
    # Get all matches where this team played
    home = df[df['home_team'] == team]
    away = df[df['away_team'] == team]
    
    # Filter by year - only use matches UP TO that year
    home = home[home['year'] <= year]
    away = away[away['year'] <= year]
    
    # Calculate attack rating
    goals_scored = home['home_score'].sum() + away['away_score'].sum()
    
    # Calculate defense rating  
    goals_conceded = home['away_score'].sum() + away['home_score'].sum()
    
    # Total matches
    total = len(home) + len(away)
    
    if total == 0:
        return None
    
    return {
        'team': team,
        'year': year,
        'attack': round(goals_scored / total, 2),
        'defense': round(goals_conceded / total, 2),
        'total_matches': total
    }
def predict_match(df, team1, year1, team2, year2):
    stats1 = calculate_team_stats(df, team1, year1)
    stats2 = calculate_team_stats(df, team2, year2)
    
    if not stats1 or not stats2:
        print("Team not found!")
        return
    
    # Calculate edge for each team
    edge1 = stats1['attack'] - stats2['defense']
    edge2 = stats2['attack'] - stats1['defense']
    
    # Convert to probability
    total = abs(edge1) + abs(edge2)
    if total == 0:
        prob1 = 0.5
    else:
        prob1 = (edge1 + total) / (2 * total)
    
    prob2 = 1 - prob1
    
    print(f"\n{'='*40}")
    print(f"{team1} ({year1}) vs {team2} ({year2})")
    print(f"{'='*40}")
    print(f"{team1} win probability: {prob1*100:.1f}%")
    print(f"{team2} win probability: {prob2*100:.1f}%")
    print(f"{'='*40}")
def build_model(df):
    print("Pre-calculating team stats...")
    
    # Build cache first - calculate each team's stats once per year
    cache = {}
    recent = df[df['year'] >= 1990].dropna(subset=['home_score', 'away_score'])
    
    teams = set(recent['home_team'].unique()) | set(recent['away_team'].unique())
    years = recent['year'].unique()
    
    for team in teams:
        for year in years:
            key = f"{team}_{year}"
            stats = calculate_team_stats(df, team, year)
            if stats:
                cache[key] = stats
    
    print("Building ML model...")
    X = []
    y1 = []
    y2 = []
    
    for _, row in recent.iterrows():
        home_stats = cache.get(f"{row['home_team']}_{row['year']}")
        away_stats = cache.get(f"{row['away_team']}_{row['year']}")
        
        if not home_stats or not away_stats:
            continue
        
        is_wc = 1 if row['tournament'] == 'FIFA World Cup' else 0
        is_neutral = 1 if row['neutral'] == True else 0
        
        features = [
            home_stats['attack'],
            home_stats['defense'],
            away_stats['attack'],
            away_stats['defense'],
            is_wc,
            is_neutral
        ]
        
        X.append(features)
        y1.append(row['home_score'])
        y2.append(row['away_score'])
    
    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    model_home = RandomForestRegressor(n_estimators=100, random_state=42)
    model_away = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model_home.fit(X, y1)
    model_away.fit(X, y2)
    
    print("Model trained successfully!")
    return model_home, model_away, cache
def predict_score(df, model_home, model_away, team1, year1, team2, year2):
    stats1 = calculate_team_stats(df, team1, year1)
    stats2 = calculate_team_stats(df, team2, year2)
    
    if not stats1 or not stats2:
        print("Team not found!")
        return
    
    features = np.array([[
        stats1['attack'],
        stats1['defense'],
        stats2['attack'],
        stats2['defense'],
        1,  # assume World Cup match
        1   # assume neutral ground
    ]])
    
    predicted_home = model_home.predict(features)[0]
    predicted_away = model_away.predict(features)[0]
    
    print(f"\n{'='*40}")
    print(f"{team1} ({year1}) vs {team2} ({year2})")
    print(f"{'='*40}")
    print(f"Predicted score: {team1} {round(predicted_home)} - {round(predicted_away)} {team2}")
    print(f"Expected goals: {predicted_home:.2f} - {predicted_away:.2f}")
    print(f"{'='*40}")
def save_model(model_home, model_away, cache):
    joblib.dump(model_home, 'model_home.pkl')
    joblib.dump(model_away, 'model_away.pkl')
    joblib.dump(cache, 'cache.pkl')
    print("Model saved!")

def load_model():
    if os.path.exists('model_home.pkl'):
        model_home = joblib.load('model_home.pkl')
        model_away = joblib.load('model_away.pkl')
        cache = joblib.load('cache.pkl')
        print("Model loaded from disk!")
        return model_home, model_away, cache
    return None, None, None
# Test it
df = load_data()

# Try loading saved model first
model_home, model_away, cache = load_model()

# If no saved model, train and save
if model_home is None:
    model_home, model_away, cache = build_model(df)
    save_model(model_home, model_away, cache)

predict_score(df, model_home, model_away, 'Belgium', 2026, 'Argentina', 2022)