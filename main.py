from fastapi import FastAPI,HTTPException,Request,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Optional,List,Dict,Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import psycopg2
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
app=FastAPI()
app = FastAPI()
# Helper to load a table from PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'Forecast',
    'user': 'postgres',
    'password': '2326'
}
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # includes HEAD
    allow_headers=["*"],
)

def load_table(table_name):
    # Create connection string
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
             f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

    # Create SQLAlchemy engine
    engine = create_engine(db_url)

    # Define and run query
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, engine)
    return df


# -------------------

# -------------
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





@app.get("/performance", include_in_schema=True)
@app.head("/performance", include_in_schema=True)
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

# --------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for forecasting"""
    # Convert Usage column
    df['Usage'] = df['Usage'].astype(str).str.replace(",", "").astype(float)
    
    # Convert Dates and set as index
    df['Date'] = pd.to_datetime(df['Dates'],format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    df = df.rename(columns={'Usage': 'y'})
    
    return df
def format_for_superset(historical_data: pd.Series, forecast_data: pd.Series, 
                       conf_int: pd.DataFrame) -> pd.DataFrame:
    """Format data for Superset visualization via Shillelagh"""
    
    # Prepare historical data
    historical_df = pd.DataFrame({
        'date': historical_data.index,
        'actual_usage': historical_data.values,
        'forecasted_usage':None,
        'lower_bound': None,
        'upper_bound': None,
        'data_type': 'actual'
    })
    
    # Prepare forecast data
    forecast_df = pd.DataFrame({
        'date': forecast_data.index,
        'actual_usage': None,
        'forecasted_usage': forecast_data.values,
        'lower_bound': conf_int.iloc[:, 0].values,
        'upper_bound': conf_int.iloc[:, 1].values,
        'data_type': 'forecast'
    })
    
    # Combine datasets
    result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
    result_df = result_df.where(pd.notnull(result_df), None)
    return result_df
def create_sarima_forecast(data: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
    """Create SARIMA forecast - train on full dataset, forecast future days"""
    if len(data) < 24:  # Need minimum data for seasonal model
        raise ValueError("Need at least 24 data points for SARIMA modeling")
    
    # Train on entire dataset
    train = data['y']
    
    # Generate future dates for forecast
    forecast_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1), 
        periods=forecast_days, 
        freq='D'
    )
    
    # Fit SARIMA model on full training data
    model = ARIMA(train, 
                  order=(2, 0, 1), 
                  seasonal_order=(0, 1, 1, 12)).fit()
    
    # Forecast future periods
    forecast_result = model.get_forecast(steps=forecast_days)
    forecast_values = forecast_result.predicted_mean
    
    # Create forecast series with future dates
    forecast = pd.Series(forecast_values.values, index=forecast_dates)
    
    # Get confidence intervals
    conf_int = forecast_result.conf_int()
    conf_int.index = forecast_dates
    
    return {
        'forecast': forecast,
        'confidence_intervals': conf_int,
        'historical_data': train
    }
    # model = auto_arima(train, seasonal=True, m=7,  # m=seasonal period (e.g., 7 for weekly seasonality)
    #                trace=True,             # Shows progress
    #                error_action='ignore',  # Ignore errors
    #                suppress_warnings=True,
    #                stepwise=True) 
    # forecast_values, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

    # # Create future dates
    # last_date = train.index[-1]
    # forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    # # Create forecast series
    # forecast = pd.Series(forecast_values, index=forecast_dates)

    # # Confidence intervals
    # conf_int_df = pd.DataFrame(conf_int, index=forecast_dates, columns=["lower_bound", "upper_bound"])

    # return {
    #     'forecast': forecast,
    #     'confidence_intervals': conf_int_df,
    #     'historical_data': train
    # }

def safe_json_convert(obj):
    """Convert numpy/pandas types to JSON-safe types"""
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
    """Generate usage forecast with configurable forecast period and table"""
    try:
        # Load and preprocess data
        raw_data = load_table("daily_mean_usage")
        processed_data = preprocess_data(raw_data)
        
        # Generate forecast
        forecast_result = create_sarima_forecast(processed_data, forecast_days)
        
        # Format for Superset
        superset_data = format_for_superset(
            forecast_result['historical_data'],
            forecast_result['forecast'],
            forecast_result['confidence_intervals']
        )
        
        # Convert to dict with JSON-safe values
        data_records = []
        for record in superset_data.to_dict('records'):
            safe_record = {k: safe_json_convert(v) for k, v in record.items()}
            data_records.append(safe_record)
        return data_records
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


