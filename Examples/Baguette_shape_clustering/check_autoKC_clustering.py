import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture

from Kmediods import merge_clusters_, KMedoids, calcualte_Distance_Matrix, Kmeans, full_dist_between_clusters, \
    full_min_dist_between_clusters

from numba import njit
@njit
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perform_clustering(model,distance_matrix,full_dist_between_clusters,combine_centers,patterns,jaccard_dist,merge_clust=1.5):
    pattern_sets = [p for p in patterns]

    # distance_matrix = np.array([[jaccard_dist(p1, p2) for p2 in pattern_sets] for p1 in pattern_sets])
    ks = model(min(len(patterns),30))

    labels=ks.fit(distance_matrix, None)
    if labels is None:
        labels = ks.predict(distance_matrix)
    medoids = ks.get_centers(pattern_sets)


    centers, final_labels, final_cluster_patterns=merge_clusters_(full_dist_between_clusters,combine_centers,labels,medoids,
                                                                  patterns,
                                                                  jaccard_dist,
                                                                  #merge_clust=2.5,
                                                                  merge_clust=merge_clust,
                                                                  verbose=True)
    return final_labels,distance_matrix,centers




def apply_clustering(model,data,raw_data,full_dist_between_clusters,ax,merge_clust=1.8):


    final_labels, distance_matrix, centers = perform_clustering(model,data,full_dist_between_clusters,model.combine_centers, raw_data, euclidean,
                                                                merge_clust=merge_clust)
    print(len(set(final_labels)))
    ax.scatter(raw_data[:, 0], raw_data[:, 1], c=final_labels, cmap='viridis', s=50)

def baguette():
    random.seed(42)
    # Parameters for the clusters
    n_samples = 100  # Number of samples per cluster
    length = 30  # Length of the baguette
    width = 0.5  # Width of the baguette (noise level)
    distance = 4  # Distance between the centroids of the two clusters

    # Generate the first baguette-shaped cluster
    x1 = np.linspace(0, length, n_samples)
    y1 = [random.random()*width+length/2 for i in x1]
    cluster1 = np.column_stack((x1, y1))

    # Generate the second baguette-shaped cluster
    x2 = np.linspace(0, length, n_samples)
    y2 = [random.random()*width+distance+length/2 for i in x2]
    cluster2 = np.column_stack((x2, y2))

    # Combine the clusters
    data = np.vstack((cluster1, cluster2))
    return data
# Anisotropicly distributed data
X=baguette()
noisy_moons = (X, X)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))

kmdeoids=KMedoids(2)
DM=calcualte_Distance_Matrix(noisy_moons[0],euclidean)

labels = kmdeoids.fit(DM, None)
ax[0][0].set_title('k=2')
ax[0][1].set_title('AutoKC with \n complete linkage (default)')
ax[0][2].set_title('AutoKC with single \n linkage and (1.8T)')

ax[0][0].set_ylim(-1,32)
ax[0][0].set_xlim(-1,32)
ax[0][1].set_ylim(-1,32)
ax[0][1].set_xlim(-1,32)
ax[0][2].set_ylim(-1,32)
ax[0][2].set_xlim(-1,32)


ax[0][0].scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], c=labels, cmap='viridis', s=50)
apply_clustering(Kmeans,noisy_moons[0],noisy_moons[0],full_dist_between_clusters,ax[0][1],merge_clust=2)
apply_clustering(Kmeans,noisy_moons[0],noisy_moons[0],full_min_dist_between_clusters,ax[0][2])

####################################################33

noisy_moons = datasets.make_moons(n_samples=200, noise=0.05, random_state=42)
kmdeoids=KMedoids(2)
DM=calcualte_Distance_Matrix(noisy_moons[0],euclidean)
labels = kmdeoids.fit(DM, None)


ax[1][0].scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], c=labels, cmap='viridis', s=50)
apply_clustering(KMedoids,DM,noisy_moons[0],full_dist_between_clusters,ax[1][1],merge_clust=2)

apply_clustering(KMedoids,DM,noisy_moons[0],full_min_dist_between_clusters,ax[1][2])

fig.text(0.05, 0.45, "KMedoids", ha="center",fontsize=12,rotation=90)

plt.show()