import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('stock_prices.csv')
prices = data['Close'].values.reshape(-1,1)

sc = MinMaxScaler()
sc_price = sc.fit_transform(prices)

def create_sequences(data, seq_l):
    X, y = [], []
    for i in range(len(data) - seq_l):
        X.append(data[i : i + seq_l])
        y.append(data[i + seq_l])
    return np.array(X), np.array(y)

seq_l = 10
X,y = create_sequences(sc_price,seq_l)

split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = Sequential([
    LSTM(50, input_shape=(seq_l, 1)),
    Dense(1)])
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)

predictions = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()