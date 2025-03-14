# MSFT-APPL-GOOG
!pip install pmdarima
# Verificar instalación de pmdarima
try:
    from pmdarima import auto_arima
    print("pmdarima instalado correctamente")
except ImportError:
    print("Error: pmdarima no está instalado. Por favor instala con !pip install pmdarima")
    raise

# Importar librerías necesarias
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

# Configurar fechas
from datetime import datetime, timedelta
end_date = datetime.now()
start_date_3d = end_date - timedelta(days=3)
start_date_10d = end_date - timedelta(days=10)

# Definir las acciones
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Función para obtener datos
def get_stock_data(tickers, start, end, interval):
    data = yf.download(tickers, start=start, end=end, interval=interval)
    if len(tickers) > 1:
        try:
            df = data['Adj Close']
        except KeyError:
            df = data['Close']
    else:
        df = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=[tickers[0]])
    elif len(tickers) > 1 and set(df.columns) != set(tickers):
        df.columns = tickers
    return df

# Obtener datos
data_5min_3days = get_stock_data(tickers, start_date_3d, end_date, '5m')
data_30min_10days = get_stock_data(tickers, start_date_10d, end_date, '30m')

# Función para prueba de raíz unitaria
def adf_test(series, title=''):
    print(f'\nPrueba ADF para {title}:')
    result = adfuller(series.dropna())
    print(f'Estadístico ADF: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("Conclusión: Serie estacionaria (rechaza H0)")
    else:
        print("Conclusión: Serie no estacionaria (no rechaza H0)")

# Función para prueba de cointegración
def cointegration_test(df):
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            print(f'\nPrueba de cointegración entre {tickers[i]} y {tickers[j]}:')
            score, p_value, _ = coint(df[tickers[i]], df[tickers[j]])
            print(f'Estadístico: {score}')
            print(f'p-valor: {p_value}')
            if p_value < 0.05:
                print("Conclusión: Hay cointegración")
            else:
                print("Conclusión: No hay cointegración")

# Función para modelos y forecasting
def analyze_time_series(df, interval_name, forecast_steps=10):
    for ticker in tickers:
        series = df[ticker].dropna()
        
        # Prueba de estacionariedad
        adf_test(series, f'{ticker} ({interval_name})')
        
        # Diferenciación si no es estacionaria
        diff_order = 0
        if adfuller(series.dropna())[1] > 0.05:
            diff_series = series.diff().dropna()
            diff_order = 1
            print(f"\nSe aplicó diferenciación de orden 1 para {ticker}")
        else:
            diff_series = series
            
        # Modelo AR
        print(f'\nModelo AR para {ticker} ({interval_name}):')
        ar_model = AutoReg(series, lags=1)
        ar_results = ar_model.fit()
        print(ar_results.summary())
        
        # Modelo ARIMA automático
        print(f'\nModelo ARIMA óptimo para {ticker} ({interval_name}):')
        auto_model = auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5, 
                              seasonal=False, stepwise=True, trace=False)
        print(f"Mejor orden ARIMA: {auto_model.order}")
        arima_results = auto_model.fit(series)
        print(arima_results.summary())
        
        # Forecasting
        forecast = auto_model.predict(n_periods=forecast_steps)
        forecast_index = pd.date_range(start=series.index[-1], 
                                     periods=forecast_steps+1, 
                                     freq='5min' if interval_name == '5min-3d' else '30min')[1:]
        
        # Visualización
        plt.figure(figsize=(12,6))
        plt.plot(series[-50:], label='Datos reales')
        plt.plot(forecast_index, forecast, label='Predicción', color='red')
        plt.title(f'Forecasting {ticker} - {interval_name}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Métricas de error del modelo
        print(f'\nMétricas para {ticker} ({interval_name}):')  # Línea corregida
        print(f'AIC: {arima_results.aic()}')
        print(f'BIC: {arima_results.bic()}')

# Análisis completo
print("=== Análisis 5 minutos - 3 días ===")
analyze_time_series(data_5min_3days, '5min-3d', forecast_steps=10)
cointegration_test(data_5min_3days)

print("\n=== Análisis 30 minutos - 10 días ===")
analyze_time_series(data_30min_10days, '30min-10d', forecast_steps=10)
cointegration_test(data_30min_10days)
