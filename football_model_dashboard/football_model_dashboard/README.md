# Football Model Dashboard

## What is included

- `dashboard_server.py`
- `football_model_refresh.py`
- `webapp/`
- `outputs/model_refresh/` training outputs and exported model artifacts
- `top5-players.csv`
- `players_data-2024_2025.csv`
- `requirements.txt`

## Run

1. Create a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
python dashboard_server.py --port 8000
```

4. Open `http://127.0.0.1:8000`.

## Notes

- The web app reads the saved CSV outputs and exported `.joblib` models in `outputs/model_refresh/`.
- `football_model_refresh.py` is included because the dashboard server imports it for shared dataset/model helpers.
- `xgboost` is listed in the requirements because that module imports it, even though the dashboard itself uses the exported sklearn models.
