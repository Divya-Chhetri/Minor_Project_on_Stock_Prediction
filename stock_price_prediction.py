import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Define the stock symbol and time frame
stock_symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2022-01-01"  # End date is set to the beginning of 2022

# Fetch historical stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Drop any missing values
stock_data.dropna(inplace=True)

# Calculate additional features
stock_data['Next Close'] = stock_data['Adj Close'].shift(-1)

# Define features (X) and target variable (y)
X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = stock_data['Next Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()
