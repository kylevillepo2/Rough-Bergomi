import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from joblib import dump
import tensorflow as tf


dataset_df = pd.read_csv('rbergomi_dataset.csv')

S0 = 1

strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30)
maturity_range = np.linspace(30 / 365.25, 2, 25)
features = ['xi0', 'eta', 'rho', 'H', 'strike', 'maturity']
target = 'implied_volatility'

X = dataset_df[features].values
y = dataset_df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_scaler = RobustScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

dump(X_scaler, "X_scaler.joblib")

model = keras.models.Sequential([
    keras.layers.Input(shape=(7,)),  
    keras.layers.Dense(64, activation='elu'),  
    keras.layers.Dense(128, activation='elu'),  
    keras.layers.Dense(64, activation='elu'),
    keras.layers.Dense(32, activation='elu'),  
    keras.layers.Dense(1, activation='linear') 
])

model.compile(optimizer='adam', loss='mae')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,  
    validation_data=(X_test_scaled, y_test),  
    epochs=100,  
    batch_size=256,  
    callbacks=[early_stopping], 
    verbose=1 
)

model.save("rbergomi_model.keras")