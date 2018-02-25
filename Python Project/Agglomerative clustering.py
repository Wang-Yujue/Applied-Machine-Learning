from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from adspy_shared_utilities import plot_labelled_scatter
from matplotlib import pyplot as plt

X, y = make_blobs(random_state=10)

cls = AgglomerativeClustering(n_clusters=3)
cls_assignment = cls.fit_predict(X)

plot_labelled_scatter(X, cls_assignment,
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])

# Creating a dendrogram
X, y = make_blobs(random_state=10, n_samples=10)
plot_labelled_scatter(X, y,
        ['Cluster 1', 'Cluster 2', 'Cluster 3'])
print(X)
# And here's the dendrogram corresponding to agglomerative clustering of the 10 points above using Ward's method.
# The index 0..9 of the points corresponds to the index of the points in the X array above.
# For example, point 0 (5.69, -9.47) and point 9 (5.43, -9.76) are the closest two points and are clustered first.
from scipy.cluster.hierarchy import ward, dendrogram
plt.figure()
dendrogram(ward(X))
plt.show()