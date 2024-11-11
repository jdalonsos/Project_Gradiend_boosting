import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the data
data = pd.read_csv('returns_data.csv')
data['returns'] = data['returns'].astype(float)

years= 15
endDate=dt.datetime.now()
startDate=endDate-dt.timedelta(days=365*years)
#Create a list  of tickers
tickers=['SPY','BND','GLD','QQQ','VTI']
#Download the daily adjusted close prices for the tickers
adj_close_df=pd.DataFrame()

for ticker in tickers:
    data=yf.download(ticker,startDate,endDate)
    adj_close_df[ticker]=data['Adj Close']

#We are gonna calculate the returns
log_returns=np.log(adj_close_df/adj_close_df.shift(1))
log_returns=log_returns.dropna()
print(log_returns)

#Create an equally weighted portfolio
portfolio_value=1000000
weights=np.array([1/len(tickers)]*len(tickers))

#Calculate the historical portfolio returns
historical_returns = (log_returns * weights).sum(axis =1)
print(historical_returns)

#Find the X day historical returns
days=5
range_returns=historical_returns.rolling(window=days).sum()
range_returns=range_returns.dropna()
print(range_returns)
#Specify a confidence itnerval and calculkate the Value at Risk using historical method
confidence_interval=0.95
VaR= -np.percentile(range_returns,100-(confidence_interval*100))*portfolio_value
print(VaR)

# Data Preparation for Machine Learning
# Define the target and features
X = log_returns.values
y = historical_returns.values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# LightGBM Model for Returns Prediction
lgb_model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
print(f"LightGBM Mean Squared Error: {mse_lgb}")

# Neural Network Model for Returns Prediction
tf.keras.backend.clear_session()
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred_nn = nn_model.predict(X_test).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f"Neural Network Mean Squared Error: {mse_nn}")

# Combine Predictions for Enhanced Accuracy
combined_pred = 0.5 * y_pred_lgb + 0.5 * y_pred_nn
mse_combined = mean_squared_error(y_test, combined_pred)
print(f"Combined Model Mean Squared Error: {mse_combined}")

# Calculate VaR using Combined Model Prediction
days = 5  # For 5-day VaR
confidence_interval = 0.95
rolling_predictions = pd.Series(combined_pred).rolling(window=days).sum().dropna()
VaR = -np.percentile(rolling_predictions, 100 - (confidence_interval * 100)) * portfolio_value
print(f"Value at Risk (VaR) using Machine Learning: {VaR}")

# Plot Results
plt.figure(figsize=(14, 7))
plt.plot(historical_returns.index[-len(y_test):], y_test, label='Actual Returns')
plt.plot(historical_returns.index[-len(y_test):], y_pred_lgb, label='LightGBM Predictions')
plt.plot(historical_returns.index[-len(y_test):], y_pred_nn, label='Neural Network Predictions')
plt.plot(historical_returns.index[-len(y_test):], combined_pred, label='Combined Predictions', linestyle='--')
plt.title('Return Predictions Using LightGBM and Neural Network')
plt.legend()
plt.show()
print("Hola")