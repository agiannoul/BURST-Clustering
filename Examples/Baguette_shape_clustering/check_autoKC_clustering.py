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
                                                                  verbose=False)
    return final_labels,distance_matrix,centers




def apply_clustering(model,data,raw_data,full_dist_between_clusters,ax,merge_clust=1.8):


    final_labels, distance_matrix, centers = perform_clustering(model,data,full_dist_between_clusters,model.combine_centers, raw_data, euclidean,
                                                                merge_clust=merge_clust)
    print(len(set(final_labels)))
    ax.scatter(raw_data[:, 0], raw_data[:, 1], c=final_labels, cmap='viridis', s=50)



noisy_moons = datasets.make_moons(n_samples=200, noise=0.05, random_state=42)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))

kmeans_model=Kmeans(2)
labels = kmeans_model.fit(noisy_moons[0], None)

ax[0][0].set_title('k=2')
ax[0][1].set_title('AutoKC with \n complete linkage (default)')
ax[0][2].set_title('AutoKC with single \n linkage and (1.8T)')
ax[0][3].set_title('AutoKC with single \n linkage and (2T)')




ax[0][0].scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], c=labels, cmap='viridis', s=50)
apply_clustering(Kmeans,noisy_moons[0],noisy_moons[0],full_dist_between_clusters,ax[0][1],merge_clust=2)
apply_clustering(Kmeans,noisy_moons[0],noisy_moons[0],full_min_dist_between_clusters,ax[0][2])
apply_clustering(Kmeans,noisy_moons[0],noisy_moons[0],full_min_dist_between_clusters,ax[0][3],merge_clust=2)


kmdeoids=KMedoids(2)
DM=calcualte_Distance_Matrix(noisy_moons[0],euclidean)
labels = kmdeoids.fit(DM, None)


ax[1][0].scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], c=labels, cmap='viridis', s=50)
apply_clustering(KMedoids,DM,noisy_moons[0],full_dist_between_clusters,ax[1][1],merge_clust=2)

apply_clustering(KMedoids,DM,noisy_moons[0],full_min_dist_between_clusters,ax[1][2])
apply_clustering(KMedoids,DM,noisy_moons[0],full_min_dist_between_clusters,ax[1][3],merge_clust=2)

fig.text(0.05, 0.65, "KMeans", ha="center",fontsize=12,rotation=90)
fig.text(0.05, 0.25, "KMedoids", ha="center",fontsize=12,rotation=90)

plt.show()