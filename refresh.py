# refresh.py  •  loads trained BayesianRidge model
"""
Run:      python refresh.py
Requires: ODDS_API_KEY in .env
          models/player_points.joblib   (trained earlier)
Output:   Top 10 player-points edges (table + JSON)
"""

import os, json, joblib, requests, pandas as pd
import re
TRAIN_LOGS = pd.read_parquet("data/player_logs.parquet")
LAST5      = pd.read_parquet("cache/last5.parquet")

def clean_name(name: str) -> str:
    """
    Normalize player names so Odds-API and nba_api match:
    • remove anything in parentheses   e.g.  "LeBron James (LAL)" → "LeBron James"
    • remove punctuation               e.g.  "J. Harden"         → "J Harden"
    • lower-case + strip spaces
    """
    name = re.sub(r"\([^)]*\)", "", name)            # strip "(…)"
    name = re.sub(r"[^A-Za-z\s]", "", name)          # keep letters & spaces
    return name.lower().strip()
from datetime import datetime, timezone
from dotenv import load_dotenv

# ── Load secrets & model ──────────────────────────────────────────────
load_dotenv()
ODDS_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_KEY:
    raise SystemExit("❌  ODDS_API_KEY missing; add it to .env")

MODEL_BUNDLE = joblib.load("models/player_points.joblib")
MODEL        = MODEL_BUNDLE["model"]
FEATS        = MODEL_BUNDLE["features"]      # ['season_pts','rolling5','home']

# season-average points per player (static lookup from training data)
TRAIN_LOGS = pd.read_parquet("data/player_logs.parquet")
SEASON_PTS = TRAIN_LOGS.groupby("PLAYER_NAME").PTS.mean()

# ── Odds API helpers ──────────────────────────────────────────────────
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"

def today_events():
    r = requests.get(f"{BASE}/events",
                     params={"apiKey": ODDS_KEY, "dateFormat": "iso"},
                     timeout=20)
    r.raise_for_status()
    return r.json()

def player_points(event_id, game_label):
    r = requests.get(f"{BASE}/events/{event_id}/odds",
                     params={"apiKey": ODDS_KEY,
                             "regions": "us",
                             "markets": "player_points"},
                     timeout=20)
    r.raise_for_status()
    rows = []
    data = r.json()
    for bk in data.get("bookmakers", []):
        m = next((m for m in bk["markets"] if m["key"] == "player_points"), None)
        if not m:
            continue
        for o in m["outcomes"]:
            rows.append({
                "player": o["description"],
                "line":   o["point"],
                "price":  o["price"],
                "game":   game_label
            })
        break
    return rows

def fetch_live_props():
    props = []
    for g in today_events():
        label = f'{g["away_team"]} @ {g["home_team"]}'
        props += player_points(g["id"], label)
    if not props:
        raise SystemExit("No player-points markets for today.")
    return pd.DataFrame(props)

# ── Build live feature frame ─────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds season_pts, rolling5, home flag.
    Uses 'clean_player' as the join key to avoid name mismatches.
    """
    # 1) create cleaned key in live prop dataframe
    df["clean_player"] = df["player"].apply(clean_name)

    # 2) prep LAST5 table with same cleaned key
    last5_clean = LAST5.copy()
    last5_clean["clean_player"] = last5_clean["PLAYER_NAME"].apply(clean_name)

    # 3) season-avg lookup (from TRAIN_LOGS)   --------------------------
    df["season_pts"] = df["clean_player"].map(
        TRAIN_LOGS.assign(clean_player=lambda t: t["PLAYER_NAME"].apply(clean_name))
                 .groupby("clean_player").PTS.mean()
    ).fillna(df["line"])   # fallback for true rookies

    # 4) merge rolling-5 averages  -------------------------------------
    df = df.merge(last5_clean[["clean_player", "rolling5_pts"]],
                  on="clean_player", how="left")

    df["rolling5"] = df["rolling5_pts"].fillna(df["season_pts"])

    # 5) home/away flag  ----------------------------------------------
    df["home"] = df["game"].str.contains(r"\s@\s").astype(int)

    return df.drop(columns=["clean_player", "rolling5_pts"])

# ── Main ─────────────────────────────────────────────────────────────
def main():
    df = fetch_live_props()
    df = add_features(df)

    df["μ"]    = MODEL.predict(df[FEATS])
    df["edge"] = df["μ"] - df["line"]
    df["conf"] = (df["edge"].abs() * 10 + 50).clip(0, 100).round().astype(int)
    df["prop"] = "PTS O " + df["line"].astype(str)

    df = df.drop_duplicates(subset=["player", "line", "game"])

    top = df.sort_values("edge", ascending=False).head(10)

    # markdown
    print(top[["player","game","prop","μ","edge","conf"]]
          .to_markdown(index=False, floatfmt=".1f"))

    # JSON
    print("\n```json")
    print(top.to_json(orient="records", indent=2))
    print("```")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\nData refreshed: {ts} UTC")
    print("\nPredictions are for informational purposes only. Bet responsibly.")

if __name__ == "__main__":
    main()
