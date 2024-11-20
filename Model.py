import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import pandas as pd

np.random.seed(123)
tf.random.set_seed(123)

class Model:
    def __init__(self):
        self.data = None
        self.model = None
    
    def __build_model(self, input_shape, outputs):
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(y_true, y_pred):
            # y_true: next day's returns, shape (batch_size, num_assets)
            # y_pred: allocations, shape (batch_size, num_assets)

            # Compute portfolio returns
            portfolio_returns = tf.reduce_sum(y_pred * y_true, axis=1)

            # Compute Sharpe ratio
            mean_return = K.mean(portfolio_returns)
            std_return = K.std(portfolio_returns)
            sharpe = mean_return / (std_return + 1e-5)

            # Return negative Sharpe to minimize
            return -sharpe

        model.compile(loss=sharpe_loss, optimizer='adam')
        return model

    def train(self, data: pd.DataFrame):
        window_size = 50

        # Calculate daily returns
        returns = data.pct_change().fillna(0)

        # Combine prices and returns
        combined_data = pd.concat([data, returns], axis=1)
        combined_data.columns = [f"{ticker}_price" for ticker in data.columns] + [f"{ticker}_return" for ticker in data.columns]

        sequences = []
        next_returns = []
        for i in range(len(combined_data) - window_size):
            seq = combined_data.iloc[i:i+window_size].values
            sequences.append(seq)
            # The next day's returns
            next_return = returns.iloc[i+window_size].values
            next_returns.append(next_return)
        sequences = np.array(sequences)
        next_returns = np.array(next_returns)

        if self.model is None:
            input_shape = (window_size, combined_data.shape[1])
            self.model = self.__build_model(input_shape, data.shape[1])

        y = next_returns  # Use next day's returns as y_true

        # Train the model with early stopping to prevent overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.model.fit(sequences, y, epochs=100, batch_size=64, verbose=1, callbacks=[early_stop])

    def predict_allocation(self, input_sequence):
        # Ensure input_sequence has the correct shape (1, window_size, num_features)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        allocation = self.model.predict(input_sequence)[0]
        # Ensure allocations sum to 1
        allocation = allocation / np.sum(allocation)
        return allocation
