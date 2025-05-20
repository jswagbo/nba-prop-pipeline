# NBA Prop Pipeline

`refresh.py` fetches NBA player prop odds from the Odds API and scores them with a trained regression model.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key**
   Create a `.env` file in the project root containing your Odds API key:
   ```bash
   ODDS_API_KEY=YOUR_KEY_HERE
   ```
   The script uses `python-dotenv` so environment variables in `.env` are loaded automatically. You may also export `ODDS_API_KEY` in your shell instead.

3. **Model and data files**
   Place the following files in the locations expected by `refresh.py`:
   - `models/player_points.joblib` – trained model bundle created during training
   - `data/player_logs.parquet` – season-long player logs used for feature lookups
   - `cache/last5.parquet` – five-game rolling averages

   Create the `models/`, `data/` and `cache/` directories if they do not already exist.

## Running

Execute:
```bash
python refresh.py
```
The script logs a table of the top edges and a JSON block with the same data.

### Log levels

Adjust the verbosity with the `LOG_LEVEL` environment variable or the
`--log-level` command line option. Valid levels are the standard Python logging
levels such as `DEBUG`, `INFO` and `WARNING`.

Examples:

```bash
LOG_LEVEL=DEBUG python refresh.py          # via environment variable
python refresh.py --log-level WARNING      # via CLI option
```
