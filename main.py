import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#Load Data
company = 'FB'

start = dt.datetime(2022,1,1)
end = dt.datetime(2023,10,13)

#data = web.DataReader(company, 'yahoo', start, end)
df = yf.download(tickers=['^GSPC'], start=start, end=end)
#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform (df['Close'].values.reshape(-1,1))

predication_days = 60

x_train = []
y_train = []

for x in range(predication_days, len(scaled_data)):
    x_train.append(scaled_data[x-predication_days:x, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))   #Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test The Model Accuracy on Existing Data '''

#Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_df = yf.download(tickers=['^GSPC'], start=test_start, end=test_end)

actual_prices = test_df['Close'].values

total_dataset = pd.concat([df['Close'], test_df['Close']], axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_df) - predication_days:].values
model_inputs = model_inputs.reshape(-1,1)

#Make prediction on Test Data

x_test = []
for x in range(predication_days, len(model_inputs)):
    x_test.append(model_inputs[x-predication_days:x, 0 ])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#Plot the Test Predications
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()

#Predict next day

real_data = [model_inputs[len(model_inputs) + 1 - predication_days:len(model_inputs+1)]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")