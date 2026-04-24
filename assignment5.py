# Problem 6: Sentiment Analysis using RNN on Network/Social Graph Text Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tensorflow import keras
from tensorflow.keras import layers

# Simulated social network graph with user reviews/posts
graph_data = {
    'User1': {'connections': ['User2', 'User3'], 'text': 'This product is amazing and works great'},
    'User2': {'connections': ['User1', 'User4'], 'text': 'Terrible experience, worst purchase ever'},
    'User3': {'connections': ['User1', 'User5'], 'text': 'Really good quality and fast delivery'},
    'User4': {'connections': ['User2', 'User5'], 'text': 'Not worth the money, very disappointing'},
    'User5': {'connections': ['User3', 'User4'], 'text': 'Excellent service and beautiful design'},
    'User6': {'connections': ['User1', 'User2'], 'text': 'Awful product broke after one day'},
    'User7': {'connections': ['User3', 'User6'], 'text': 'Loved it highly recommend to everyone'},
    'User8': {'connections': ['User4', 'User7'], 'text': 'Poor quality do not buy this item'},
}

# Build and visualize network graph
G = nx.Graph()
for user, data in graph_data.items():
    G.add_node(user)
    for conn in data['connections']:
        G.add_edge(user, conn)

# Use IMDB dataset for training the RNN sentiment model
max_features = 10000
maxlen = 100
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# Build RNN model
model = keras.Sequential([
    layers.Embedding(max_features, 64, input_length=maxlen),
    layers.SimpleRNN(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Tokenize graph user texts and predict sentiment
word_index = keras.datasets.imdb.get_word_index()
def encode_text(text):
    tokens = [word_index.get(w.lower(), 0) + 3 for w in text.split()]
    return keras.preprocessing.sequence.pad_sequences([tokens], maxlen=maxlen)

# Predict sentiments for graph nodes
print("\n--- Sentiment Analysis on Network Graph ---")
sentiments = {}
for user, data in graph_data.items():
    encoded = encode_text(data['text'])
    score = model.predict(encoded, verbose=0)[0][0]
    label = 'Positive' if score > 0.5 else 'Negative'
    sentiments[user] = label
    print(f"{user}: \"{data['text']}\" -> {label} ({score:.4f})")

# Visualize graph with sentiment colors
colors = ['green' if sentiments[n] == 'Positive' else 'red' for n in G.nodes()]
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color=colors, node_size=1500, font_size=10, font_weight='bold')
plt.title("Network Graph - Sentiment Analysis (Green=Positive, Red=Negative)")
plt.show()
