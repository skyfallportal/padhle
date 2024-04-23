import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

data = pd.read_csv('boston_house_prices_f - boston_house_prices.csv.csv')

X = data.drop('MEDV',axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

model = Sequential([Dense(64, activation='relu', input_shape=(X_train_sc.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(1)])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train_sc, y_train, epochs=50, batch_size=32, validation_data=(X_test_sc, y_test))

y_pred = model.predict(X_test_sc)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()