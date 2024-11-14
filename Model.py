import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import pandas as pd

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

    def get_allocations(self, data: pd.DataFrame):
        window_size = 50

        # Combine prices and returns
        prices = data.values
        returns = data.pct_change().fillna(0).values
        combined_data = np.concatenate([prices, returns], axis=1)

        sequences = []
        next_returns = []
        for i in range(len(combined_data) - window_size):
            seq = combined_data[i:i+window_size]
            sequences.append(seq)
            # The next day's returns
            next_return = returns[i+window_size]
            next_returns.append(next_return)
        sequences = np.array(sequences)
        next_returns = np.array(next_returns)

        if self.model is None:
            input_shape = (window_size, combined_data.shape[1])
            self.model = self.__build_model(input_shape, data.shape[1])

        y = next_returns  # Use next day's returns as y_true

        # Train the model
        self.model.fit(sequences, y, epochs=100, batch_size=64, verbose=1)

        # Predict using the last available sequence
        last_sequence = combined_data[-window_size:]
        allocation = self.model.predict(last_sequence[np.newaxis, :])[0]
        return allocation

