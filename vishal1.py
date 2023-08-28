import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
yf.pdr_override()

# Define the stock to be used
stock_name = 'AAPL'  # Apple Inc.

# Define the date range
start_date = '2013-01-01'
end_date = '2023-05-22'

# Get the stock data
df = pdr.get_data_yahoo(stock_name, start=start_date, end=end_date)

# Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on (80% of the data)
training_data_len = math.ceil(len(dataset) * .8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
 
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
 
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f'RMSE: {rmse}')

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Number of trading days in a year is approximately 252
forecast_days = 5 * 252  

# Use the last 60 days of the current data as the initial input for prediction
input_data = scaled_data[-60:]

for i in range(forecast_days):
    # Reshape and expand dimensions to match the input shape for LSTM
    input_data_reshaped = np.expand_dims(input_data[-60:], axis=0)

    # Make prediction for the next day
    prediction = model.predict(input_data_reshaped)

    # Append the new prediction to the input data
    input_data = np.append(input_data, prediction, axis=0)

# Reverse the normalization of the predicted data
forecasted_data = scaler.inverse_transform(input_data[60:])

# Create a pandas date range for the forecasted period
forecast_period = pd.date_range(start=df.index[-1], periods=forecast_days, freq='B')

# Create a pandas dataframe for the forecasted data
forecast_df = pd.DataFrame(forecasted_data, index=forecast_period, columns=['Forecast'])

# Plot the actual and forecasted data
plt.figure(figsize=(16, 8))
plt.title('Forecast for the next 5 years')
plt.plot(df['Close'])
plt.plot(forecast_df['Forecast'])
plt.legend(['Actual', 'Forecast'], loc='lower right')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
