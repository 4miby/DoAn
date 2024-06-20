from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Flatten
from tensorflow.keras.models import Model

class DecisionNetwork:
    def __init__(self, input_dim, embedding_dim, hidden_dim, action_size):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_size = action_size

    def build_model(self):
        inputs = Input(shape=(None,), dtype='int32')
        x = Embedding(self.input_dim, self.embedding_dim)(inputs)
        x = LSTM(self.hidden_dim, return_sequences=True)(x)
        x = LSTM(self.hidden_dim)(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs, outputs)
        return model
