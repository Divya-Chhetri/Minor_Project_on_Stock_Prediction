# Stock Price Prediction Project

This repository contains a Python project for predicting stock prices using historical data fetched from Yahoo Finance and machine learning techniques.

## Introduction

Stock price prediction is a crucial task for investors and traders to make informed decisions. This project aims to develop a predictive model using machine learning algorithms to forecast stock prices based on historical data.

## Features

- Fetch historical stock data using the Yahoo Finance API.
- Preprocess data by dropping missing values and calculating additional features.
- Train a Linear Regression model to predict future stock prices.
- Evaluate model performance using test data and visualize predictions.

## Setup

To run the project locally, follow these steps:

1. Clone this repository:
       git clone https://github.com/Divya-Chhetri/stock-price-prediction.git


2. Install the required dependencies:
       pip install -r requirements.txt


3. Run the main script:
       python stock_price_prediction.py


## Usage

- Modify the `stock_symbol`, `start_date`, and `end_date` variables in the script to fetch data for a different stock and time frame.
- Adjust the features used for prediction by modifying the `X` variable.
- Experiment with different machine learning models for prediction by replacing `LinearRegression()` with other models from scikit-learn.

## Results

After running the script, you will see a plot displaying the actual and predicted stock prices. Evaluate the model's performance based on metrics such as Mean Squared Error and R-squared.

## Contribution

Contributions to this project are welcome! If you have any ideas for improvement or would like to report issues, feel free to open an issue or submit a pull request.

## Acknowledgements

- This project utilizes the Yahoo Finance API for fetching historical stock data.
- Special thanks to the scikit-learn and matplotlib libraries for machine learning and visualization capabilities.
