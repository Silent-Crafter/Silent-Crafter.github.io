# Problem 3: RNN/LSTM for Sentiment Analysis on IMDB Reviews
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load IMDB dataset (already tokenized)
max_features = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Pad sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(max_features, 64, input_length=maxlen),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Predict on sample
sample = X_test[:5]
preds = model.predict(sample)
for i, p in enumerate(preds):
    print(f"Review {i+1}: {'Positive' if p > 0.5 else 'Negative'} (score: {p[0]:.4f})")
