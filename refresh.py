# refresh.py  •  loads trained BayesianRidge model
"""
Run:      python refresh.py
Requires: ODDS_API_KEY in .env
          models/player_points.joblib   (trained earlier)
Output:   Top 20 player-points edges (table + JSON)
"""

import os, joblib, requests, pandas as pd
import re
import logging
import argparse
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    TRAIN_LOGS = pd.read_parquet("data/player_logs.parquet")
except FileNotFoundError as e:
    raise SystemExit("❌  data/player_logs.parquet missing; ensure the training data exists.") from e

try:
    LAST5 = pd.read_parquet("cache/last5.parquet")
except FileNotFoundError as e:
    raise SystemExit("❌  cache/last5.parquet missing; generate the rolling-5 averages first.") from e

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
# ── Load secrets & model ──────────────────────────────────────────────
load_dotenv()
ODDS_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_KEY:
    raise SystemExit("❌  ODDS_API_KEY missing; add it to .env")

# Load models for points, rebounds, and assists
try:
    POINTS_MODEL_BUNDLE = joblib.load("models/player_points.joblib")
    REBOUNDS_MODEL_BUNDLE = joblib.load("models/player_rebounds.joblib")
    ASSISTS_MODEL_BUNDLE = joblib.load("models/player_assists.joblib")
except FileNotFoundError as e:
    raise SystemExit("❌  One or more model files are missing in the models/ directory.") from e

POINTS_MODEL, POINTS_FEATS = POINTS_MODEL_BUNDLE["model"], POINTS_MODEL_BUNDLE["features"]
REBOUNDS_MODEL, REBOUNDS_FEATS = REBOUNDS_MODEL_BUNDLE["model"], REBOUNDS_MODEL_BUNDLE["features"]
ASSISTS_MODEL, ASSISTS_FEATS = ASSISTS_MODEL_BUNDLE["model"], ASSISTS_MODEL_BUNDLE["features"]

# season-average points per player (static lookup from training data)
TRAIN_LOGS = pd.read_parquet("data/player_logs.parquet")
SEASON_PTS = TRAIN_LOGS.groupby("PLAYER_NAME").PTS.mean()

# ── Odds API helpers ──────────────────────────────────────────────────
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"

def today_events():
    """Return today's NBA events from the Odds API as raw JSON."""
    r = requests.get(f"{BASE}/events",
                     params={"apiKey": ODDS_KEY, "dateFormat": "iso"},
                     timeout=20)
    r.raise_for_status()
    return r.json()

def player_props(event_id: str, game_label: str, market: str) -> list[dict]:
    """Return player prop lines for a specific market.

    Parameters
    ----------
    event_id: str
        Odds API event identifier.
    game_label: str
        Human readable game label.
    market: str
        Market key such as ``"player_points"`` or ``"player_rebounds"``.

    Returns
    -------
    list[dict]
        Each entry has ``player``, ``line``, ``price``, ``game`` and ``market``
        keys.
    """
    r = requests.get(f"{BASE}/events/{event_id}/odds",
                     params={"apiKey": ODDS_KEY,
                             "regions": "us",
                             "markets": market},
                     timeout=20)
    r.raise_for_status()
    rows = []
    data = r.json()
    for bk in data.get("bookmakers", []):
        m = next((m for m in bk["markets"] if m["key"] == market), None)
        if not m:
            continue
        for o in m.get("outcomes", []):
            rows.append({
                "player": o["description"],
                "line":   o["point"],
                "price":  o["price"],
                "game":   game_label,
                "market": market,
            })
        break
    return rows


def player_points(event_id: str, game_label: str) -> list[dict]:
    """Wrapper for ``player_props`` using the ``player_points`` market."""
    return player_props(event_id, game_label, "player_points")


def player_rebounds(event_id: str, game_label: str) -> list[dict]:
    """Wrapper for ``player_props`` using the ``player_rebounds`` market."""
    return player_props(event_id, game_label, "player_rebounds")


def player_assists(event_id: str, game_label: str) -> list[dict]:
    """Wrapper for ``player_props`` using the ``player_assists`` market."""
    return player_props(event_id, game_label, "player_assists")

def fetch_live_props() -> pd.DataFrame:
    """Collect player prop markets for today's games as a DataFrame."""
    props: list[dict] = []
    for g in today_events():
        label = f'{g["away_team"]} @ {g["home_team"]}'
        props += player_points(g["id"], label)
        props += player_rebounds(g["id"], label)
        props += player_assists(g["id"], label)
    if not props:
        raise SystemExit("No player prop markets for today.")
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

    all_predictions = []

    # Generate predictions for each market
    for market, model, feats in [
        ("player_points", POINTS_MODEL, POINTS_FEATS),
        ("player_rebounds", REBOUNDS_MODEL, REBOUNDS_FEATS),
        ("player_assists", ASSISTS_MODEL, ASSISTS_FEATS),
    ]:
        market_df = df[df["market"] == market].copy()

        if not market_df.empty:
            market_df["μ"] = model.predict(market_df[feats])
            market_df["edge"] = market_df["μ"] - market_df["line"]
            market_df["conf"] = (market_df["edge"].abs() * 10 + 50).clip(0, 100).round().astype(int)
            market_df["prop"] = f"{market.split('_')[1].upper()} O " + market_df["line"].astype(str)

            all_predictions.append(market_df)

    if not all_predictions:
        logging.info("No predictions generated for any market.")
        return

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    top = all_predictions_df.sort_values("edge", ascending=False).head(20)

    # markdown
                 top.to_json(orient="records", indent=2))

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logging.info("\nData refreshed: %s UTC", ts)
    logging.info("\nPredictions are for informational purposes only. Bet responsibly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level",
                        default=os.getenv("LOG_LEVEL", "INFO"),
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s",
                        stream=sys.stdout)

    main()
