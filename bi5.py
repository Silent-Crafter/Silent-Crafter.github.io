from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = load_iris()
X = data.data

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_

print("K-Means Clustering Results")
print("=" * 40)
print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
print(f"\nCluster Labels: {labels}")
print(f"Inertia: {kmeans.inertia_:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.savefig('clustering_output.png')
plt.show()
print("Visualization saved as clustering_output.png")
