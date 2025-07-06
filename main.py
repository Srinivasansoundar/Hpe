from fastapi import FastAPI, HTTPException, Query,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

class ADWIN:
    """
    Optimized Adaptive Windowing (ADWIN) algorithm for drift detection
    """
    def __init__(self, delta: float = 0.005, min_window_size: int = 10):
        self.delta = delta
        self.min_window_size = min_window_size
        self.bucket_list = []
        self.width = 0
        self.total = 0.0
        self._mean = 0.0
        self._variance = 0.0
        
    def add_element(self, value: float) -> bool:
        """Add new element and check for drift"""
        if pd.isna(value):
            return False
            
        self.bucket_list.append(value)
        self.width += 1
        
        # Incremental mean calculation
        old_mean = self._mean
        self._mean = old_mean + (value - old_mean) / self.width
        
        # Incremental variance calculation (Welford's algorithm)
        if self.width == 1:
            self._variance = 0.0
        else:
            self._variance = ((self.width - 1) * self._variance + 
                            (value - old_mean) * (value - self._mean)) / self.width
        
        return self._detect_change()
    
    def _detect_change(self) -> bool:
        """Optimized drift detection"""
        if self.width < self.min_window_size * 2:
            return False
        
        # Use sliding window approach for better performance
        best_cut = self._find_best_cut_point()
        
        if best_cut is None:
            return False
        
        # Reset window to recent data if drift detected
        keep_size = self.width - best_cut
        self.bucket_list = self.bucket_list[best_cut:]
        self.width = len(self.bucket_list)
        
        # Recalculate statistics
        if self.width > 0:
            self._mean = np.mean(self.bucket_list)
            self._variance = np.var(self.bucket_list, ddof=1) if self.width > 1 else 0.0
        else:
            self._mean = 0.0
            self._variance = 0.0
        
        return True
    
    def _find_best_cut_point(self) -> Optional[int]:
        """Find the best cut point using ADWIN criterion"""
        n = self.width
        
        # Check multiple cut points for better accuracy
        for i in range(self.min_window_size, n - self.min_window_size + 1):
            left_window = self.bucket_list[:i]
            right_window = self.bucket_list[i:]
            
            if len(left_window) < self.min_window_size or len(right_window) < self.min_window_size:
                continue
            
            # Calculate statistics
            mean_left = np.mean(left_window)
            mean_right = np.mean(right_window)
            var_left = np.var(left_window, ddof=1) if len(left_window) > 1 else 0
            var_right = np.var(right_window, ddof=1) if len(right_window) > 1 else 0
            
            n1, n2 = len(left_window), len(right_window)
            
            # Combined variance
            if n1 > 1 and n2 > 1:
                variance_combined = ((n1 - 1) * var_left + (n2 - 1) * var_right) / (n1 + n2 - 2)
            else:
                variance_combined = max(var_left, var_right)
            
            if variance_combined <= 0:
                # Use simple difference test
                if abs(mean_left - mean_right) > 1e-6:
                    return i
                continue
            
            # ADWIN test statistic with corrected formula
            harmonic_mean = 2 / (1/n1 + 1/n2)
            epsilon_cut = np.sqrt((2 * variance_combined * np.log(2 / self.delta)) / harmonic_mean)
            
            if abs(mean_left - mean_right) > epsilon_cut:
                return i
        
        return None
def remove_seasonality(df: pd.DataFrame, metric: str, freq: int = 30) -> pd.DataFrame:
    """
    Optimized seasonality removal with better error handling
    """
    df = df.copy()
    
    # Ensure metric column is numeric
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    
    # Sort by datetime without setting as index (more efficient)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Remove rows with NaN values in the metric
    valid_data = df.dropna(subset=[metric])
    
    if len(valid_data) < freq * 3:
        logger.warning(f"Insufficient data for seasonality removal: {len(valid_data)} < {freq * 3}")
        df['residual'] = df[metric]
        return df
    
    try:
        # Create a proper time series
        ts = valid_data.set_index('datetime')[metric]
        
        # Use seasonal_decompose with optimized parameters
        decomposition = seasonal_decompose(
            ts,
            model='additive',
            period=freq,
            extrapolate_trend='freq',
            two_sided=False
        )
        
        # Check decomposition quality
        residuals = decomposition.resid
        nan_ratio = residuals.isna().sum() / len(residuals)
        
        if nan_ratio > 0.3:  # More than 30% NaN
            logger.warning(f"Poor decomposition quality ({nan_ratio:.1%} NaN). Using original data.")
            df['residual'] = df[metric]
        else:
            # Map residuals back to original dataframe
            residual_dict = residuals.to_dict()
            df['residual'] = df['datetime'].map(residual_dict).fillna(df[metric])
            
            # Log decomposition quality
            correlation = np.corrcoef(
                df[metric].dropna(),
                df['residual'].dropna()
            )[0, 1] if len(df.dropna()) > 1 else 1.0
            
            logger.info(f"Seasonality removal correlation: {correlation:.3f}")
            
    except Exception as e:
        logger.error(f"Seasonality removal failed: {e}. Using original data.")
        df['residual'] = df[metric]
    
    return df


def detect_drift_with_adwin(df: pd.DataFrame, metric: str, delta: float = 0.002) -> List[datetime]:
    """
    Optimized drift detection using ADWIN
    """
    if len(df) < 20:
        logger.warning(f"Insufficient data for ADWIN: {len(df)} points")
        return []
    
    # Sort data by datetime
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    adwin = ADWIN(delta=delta, min_window_size=10)
    drift_dates = []
    
    logger.info(f"Running ADWIN on {len(df_sorted)} data points")
    
    for _, row in df_sorted.iterrows():
        value = row[metric]
        if pd.notna(value):
            drift_detected = adwin.add_element(float(value))
            
            if drift_detected:
                drift_date = row['datetime']
                drift_dates.append(drift_date)
                logger.info(f"ADWIN drift detected at {drift_date.date()}")
    
    return drift_dates
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Usage'] = df['Usage'].astype(str).str.replace(",", "").astype(float)
    df['Date'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
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
        raw_data = load_table("daily_mean_usage")
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
        raw_data['Usage'] = raw_data['Usage'].astype(str).str.replace(",", "").astype(float)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d-%m-%Y')
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





@app.get("/trend_drift")
@app.head("/trend_drift")
async def detect_residual_drift(
    start: str = Query(..., description="Start date for drift detection (YYYY-MM-DD)"),
    end: str = Query(..., description="End date for drift detection (YYYY-MM-DD)")
):
    try:
        # Step 1: Load & preprocess data
        raw_data = load_table("daily_mean_usage")
        raw_data['Usage'] = raw_data['Usage'].astype(str).str.replace(",", "").astype(float)
        raw_data['Date'] = pd.to_datetime(raw_data['Dates'], format='%d-%m-%Y')
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

        # Step 4: Apply Page-Hinkley on trend
        ph = PageHinkley(min_instances=30, delta=1.0, threshold=5.0)
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


@app.get("/residual_drift", include_in_schema=True)
@app.head("/residual_drift", include_in_schema=True)
async def detect_drift(
    request: Request,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    delta: float = Query(0.005, description="ADWIN confidence parameter (lower = more sensitive)")
) -> List[Dict[str, Any]]:
    """
    Optimized drift detection endpoint with usage values included
    """
    if request.method == "HEAD":
        return JSONResponse(content={})
    
    try:
        # Load and prepare data
        table_name = "daily_mean_usage"
        df = load_table(table_name)
        
        if df.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No data found in table"}
            )
        
        # Data preprocessing
        df['datetime'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
        df['date'] = df['datetime'].dt.date
        
        logger.info(f"Loaded {len(df)} data points from {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        # Determine date range
        if not start_date or not end_date:
            latest_date = df['datetime'].max().date()
            if not end_date:
                end_date = latest_date.strftime('%Y-%m-%d')
            if not start_date:
                start_date = (latest_date - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Filter by date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        
        df_filtered = df[
            (df['datetime'] >= start_dt) & (df['datetime'] < end_dt)
        ].copy()
        
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Filtered data: {len(df_filtered)} points")
        
        if len(df_filtered) < 20:
            return JSONResponse(
                status_code=400,
                content={"error": f"Insufficient data: {len(df_filtered)} points. Need at least 20 days."}
            )
        
        # Remove seasonality
        df_processed = remove_seasonality(df_filtered, 'Usage')
        
        if 'residual' not in df_processed.columns:
            return JSONResponse(
                status_code=500,
                content={"error": "Seasonality removal failed"}
            )
        
        # Detect drift
        drift_dates = detect_drift_with_adwin(df_processed, 'residual', delta)
        drift_date_set = {dt.date() for dt in drift_dates}
        
        # Prepare output data with usage values
        output_data = []
        
        # Group by date to handle multiple entries per day
        daily_data = df_processed.groupby('date').agg({
            'Usage': 'mean',  # Average usage for the day
            'residual': 'mean'  # Average residual for the day
        }).reset_index()
        
        for _, row in daily_data.iterrows():
            date = row['date']
            date_str = date.strftime('%Y-%m-%d')
            usage_value = float(row['Usage']) if pd.notna(row['Usage']) else None
            drift_detected = date in drift_date_set
            
            output_data.append({
                "date": date_str,
                "metric": "Usage",
                "usage": usage_value,
                "drift_detected": drift_detected,
                "drift_score": None  # ADWIN doesn't provide continuous scores
            })
        
        # Sort by date for consistent output
        output_data.sort(key=lambda x: x['date'])
        
        logger.info(f"Returning {len(output_data)} data points, drift detected on {len(drift_date_set)} dates")
        return output_data
        
    except ValueError as e:
        logger.error(f"Invalid input parameters: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Invalid parameters: {str(e)}"})
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
