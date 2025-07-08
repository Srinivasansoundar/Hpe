<h1 align="center" id="title">Time Series Forecasting &amp; Drift Detection – FastAPI Service</h1>

<p id="description">This FastAPI-based service enables time series forecasting and drift detection through RESTful APIs. It uses SARIMA to generate forecasts while capturing both seasonal and trend components in the data. For drift detection it leverages Page-Hinkley and ADWIN algorithms to identify changes in usage patterns over time. Fast Fourier Transform (FFT) is used during preprocessing to estimate the seasonality period. The service is optimized for integration with visualization tools like Apache Superset using Shillelagh for real-time dashboarding.</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the Repository</p>

```
git clone https://github.com/Srinivasansoundar/Hpe.git
```

<p>2. Install Dependencies</p>

```
pip install -r requirement.txt
```

<p>3. Run the FastAPI Service</p>

```
uvicorn main:app --reload
```

<p>4. The server will start at</p>

```
http://127.0.0.1:8000
```

<p>5. Available Endpoints</p>

```
/forecast?days=30 
```

```
/trend_drift?start=YYYY-MM-DD&end=YYYY-MM-DD
```
