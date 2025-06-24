from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft
from scipy.spatial.distance import jensenshannon
from river.drift import PageHinkley

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
load_dotenv()  # Load variables from .env into environment

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def load_table(table_name):
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
             f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    engine = create_engine(db_url)
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, engine)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Usage'] = df['usage'].astype(str).str.replace(",", "").astype(float)
    df['Date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    return df.rename(columns={'Usage': 'y'})

def format_for_superset(historical_data: pd.Series, forecast_data: pd.Series, 
                       conf_int: pd.DataFrame) -> pd.DataFrame:
    historical_df = pd.DataFrame({
        'date': historical_data.index,
        'actual_usage': historical_data.values,
        'forecasted_usage': None,
        'lower_bound': None,
        'upper_bound': None,
        'data_type': 'actual'
    })
    forecast_df = pd.DataFrame({
        'date': forecast_data.index,
        'actual_usage': None,
        'forecasted_usage': forecast_data.values,
        'lower_bound': conf_int.iloc[:, 0].values,
        'upper_bound': conf_int.iloc[:, 1].values,
        'data_type': 'forecast'
    })
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
    return result_df.where(pd.notnull(result_df), None)

def create_sarima_forecast(data: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
    if len(data) < 24:
        raise ValueError("Need at least 24 data points for SARIMA modeling")
    train = data['y']
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), 
                                   periods=forecast_days, freq='D')
    model = ARIMA(train, order=(2, 0, 1), seasonal_order=(0, 1, 1, 12)).fit()
    forecast_result = model.get_forecast(steps=forecast_days)
    forecast = pd.Series(forecast_result.predicted_mean.values, index=forecast_dates)
    conf_int = forecast_result.conf_int()
    conf_int.index = forecast_dates
    return {
        'forecast': forecast,
        'confidence_intervals': conf_int,
        'historical_data': train
    }

def safe_json_convert(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

@app.get("/forecast")
@app.head("/forecast")
async def get_forecast(
    forecast_days: int = Query(30, description="Number of days to forecast", ge=1, le=365)
):
    try:
        raw_data = load_table("daily_usage")
        processed_data = preprocess_data(raw_data)
        forecast_result = create_sarima_forecast(processed_data, forecast_days)
        superset_data = format_for_superset(
            forecast_result['historical_data'],
            forecast_result['forecast'],
            forecast_result['confidence_intervals']
        )
        return [{k: safe_json_convert(v) for k, v in record.items()} for record in superset_data.to_dict('records')]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


    
@app.get("/seasonal_drift")
@app.head("/seasonal_drift")
async def detect_seasonal_drift(
    start: str = Query(..., description="Start date for current window (YYYY-MM-DD)"),
    end: str = Query(..., description="End date for current window (YYYY-MM-DD)"),
    threshold: float = Query(0.3, description="JS divergence threshold to detect drift")
):
    try:
        # Load and preprocess
        raw_data = load_table("daily_usage")
        raw_data['Usage'] = raw_data['usage'].astype(str).str.replace(",", "").astype(float)
        raw_data['Date'] = pd.to_datetime(raw_data['date'], format='%d-%m-%Y')
        raw_data = raw_data.sort_values('Date')
        raw_data['Usage'] = raw_data['Usage'].interpolate()

        # Seasonal decomposition
        decomp = seasonal_decompose(raw_data['Usage'], model='additive', period=30)
        raw_data['Seasonal'] = decomp.seasonal.interpolate()
        raw_data.dropna(subset=['Seasonal'], inplace=True)

        # Parse date inputs
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        cur_days = (end_dt - start_dt).days + 1
        ref_start = start_dt - timedelta(days=cur_days)
        ref_end = start_dt - timedelta(days=1)

        # Helper: FFT Distribution
        def get_fft_distribution(x, n_components=20):
            fft_vals = fft(x)
            magnitudes = np.abs(fft_vals)[:n_components]
            return magnitudes / np.sum(magnitudes)

        # Get FFTs
        ref_vals = raw_data[(raw_data['Date'] >= ref_start) & (raw_data['Date'] <= ref_end)]['Seasonal'].values
        cur_vals = raw_data[(raw_data['Date'] >= start_dt) & (raw_data['Date'] <= end_dt)]['Seasonal'].values

        if len(ref_vals) < 30 or len(cur_vals) < 30:
            raise HTTPException(status_code=400, detail="Not enough data for drift detection")

        ref_fft = get_fft_distribution(ref_vals)
        cur_fft = get_fft_distribution(cur_vals)
        js_div = jensenshannon(ref_fft, cur_fft)
        drift_detected = js_div > threshold

        # Create output format
        results = []

        for _, row in raw_data[(raw_data['Date'] >= ref_start) & (raw_data['Date'] <= ref_end)].iterrows():
            results.append({
                "timestamp": row['Date'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                "usage": round(row['Seasonal'], 2),
                "original_usage": round(row['Usage'], 2),  # ⬅️ This line added
                "component": "Seasonal",
                "window_type": "reference",
                "drift_detected": False,
                "drift_score": round(js_div, 4)
            })



        for _, row in raw_data[(raw_data['Date'] >= start_dt) & (raw_data['Date'] <= end_dt)].iterrows():
            results.append({
                "timestamp": row['Date'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                "usage": round(row['Seasonal'], 2),
                "original_usage": round(row['Usage'], 2),  # ⬅️ This line added
                "component": "Seasonal",
                "window_type": "current",
                "drift_detected": bool(drift_detected),
                "drift_score": round(js_div, 4)
            })



        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/residual_drift")
@app.head("/residual_drift")
async def detect_residual_drift(
    start: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end: str = Query(..., description="End date (YYYY-MM-DD)")
):
    try:
        # Load & clean data
        raw_data = load_table("daily_usage_interpolated")
        raw_data['Usage'] = raw_data['usage'].astype(str).str.replace(",", "").astype(float)
        raw_data['Date'] = pd.to_datetime(raw_data['date'], format='%d-%m-%Y')
        raw_data.sort_values('Date', inplace=True)
        raw_data['Usage'] = raw_data['Usage']
        

        # Seasonal decomposition
        decomp = seasonal_decompose(raw_data['Usage'], model='additive', period=30)
        raw_data['Residual'] = decomp.resid
        raw_data.dropna(subset=['Residual'], inplace=True)

        # Create drift flag column
        raw_data['drift_detected'] = False

        # Define detection window
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        window_data = raw_data[(raw_data['Date'] >= start_dt) & (raw_data['Date'] <= end_dt)]

        # Apply Page-Hinkley
        ph = PageHinkley(min_instances=30, delta=0.1, threshold=0.1)
        for idx in window_data.index:
            ph.update(raw_data.at[idx, 'Residual'])
            if ph.drift_detected:
                raw_data.at[idx, 'drift_detected'] = True

        # Format output
        results = []
        for row in raw_data.itertuples(index=False):
            results.append({
                "timestamp": row.Date.isoformat(),
                "original_usage": float(round(row.Usage, 9)),
                "residual": float(round(row.Residual, 2)),
                "component": "Residual",
                "drift_detected": bool(row.drift_detected)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/trend_drift")
@app.head("/trend_drift")
async def detect_residual_drift(
    start: str = Query(..., description="Start date for drift detection (YYYY-MM-DD)"),
    end: str = Query(..., description="End date for drift detection (YYYY-MM-DD)")
):
    try:
        # Step 1: Load & preprocess data
        raw_data = load_table("daily_usage_interpolated")
        raw_data['Usage'] = raw_data['usage'].astype(str).str.replace(",", "").astype(float)
        raw_data['Date'] = pd.to_datetime(raw_data['date'], format='%d-%m-%Y')
        raw_data.sort_values('Date', inplace=True)
        raw_data['Usage'] = raw_data['Usage']

        # Step 2: Seasonal decomposition
        decomp = seasonal_decompose(raw_data['Usage'], model='additive', period=30)
        raw_data['Trend'] = decomp.trend
        raw_data.dropna(subset=['Trend'], inplace=True)

        # Step 3: Define window
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        window_data = raw_data[(raw_data['Date'] >= start_dt) & (raw_data['Date'] <= end_dt)]

        # Step 4: Apply Page-Hinkley on residuals
        ph = PageHinkley(min_instances=30, delta=0.1, threshold=0.1)
        drift_flags = []
        for val in window_data['Trend']:
            ph.update(val)
            drift_flags.append(ph.drift_detected)

        # Step 5: Format results
        results = []
        for i, (idx, row) in enumerate(window_data.iterrows()):
            results.append({
                "timestamp": row['Date'].strftime('%Y-%m-%d'),
                "original_usage": float(round(row['Usage'], 5)),
                "trend": float(round(row['Trend'], 2)),
                "component": "Trend",
                "drift_detected": bool(drift_flags[i])
            })


        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
