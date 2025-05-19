import statistics

import numpy as np
from sklearn.cluster import KMeans as KMeanscore


class Kmeans():
    def __init__(self,k=2):
        self.k=k
        self.model=KMeanscore(n_clusters=self.k, random_state=42, n_init="auto")
    def fit(self, X, y):
        self.model = self.model.fit(X)
        self.cluster_centers_=self.model.cluster_centers_

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def combine_centers(list_indx_centers,cluster_subseq,cluster_centers,dist_function):
        """
        Combine two centers given their sub-sequences (this is called from AutoK)
        """
        all_sub_to_add = []
        for lb in list_indx_centers:
            all_sub_to_add.extend(cluster_subseq[lb])

        mean_vector = np.mean(np.array(all_sub_to_add), axis=0)
        return mean_vector,all_sub_to_add
    def get_centers(self,X):
        return self.cluster_centers_
    @staticmethod
    def meta_store(meta_data,cluster_subseq,cluster_center,name,distance_function):
        """
            Store meta date, for kmedoids we store the number of subsequences in each cluster (count)
        """
        if name not in meta_data:
            meta_data[name] = {}
        meta_data[name]['count'] = len(cluster_subseq)
        return meta_data

    @staticmethod
    def combine_old_new(meta_data,name,new_subs,old_center,new_centers,dist_function):
        """
            Combine old clusters (from previous batches with new clusters from current batch)
        """
        # Merge sizes
        new_center=np.array(new_subs).mean(axis=0)
        merged_size = meta_data[name]["count"] + len(new_subs)
        #weighted sum
        merged_centroid = (meta_data[name]["count"] *  old_center + len(new_subs) * new_center) / merged_size
        meta_data[name]["count"]=merged_size
        return merged_centroid,meta_data



    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }



class KMedoids():

    def __init__(self, num_clusters, max_T=100):
        self.num_clusters = num_clusters
        self.max_T = max_T
        self.num_iter = 0
        self.medoids = None
        self._labels = None
    def get_centers(self,pattern_sets):
        medoids=[]
        for i in self.medoids:
            medoids.append(pattern_sets[i])
        return medoids
    def _init(self, DM):
        np.random.seed(123)
        self.medoids = np.random.choice(range(DM.shape[0]),
            self.num_clusters, replace=False)

    def _find_labels(self, DM):
        labels = self.medoids[np.argmin(DM[:,self.medoids], axis=1)]
        labels[self.medoids] = self.medoids # ensure medoids are labelled to themselves
        return labels

    def _find_medoids(self, labels, DM):
        new_medoids = np.array([-1] * self.num_clusters)
        for medoid in self.medoids:
            cluster = np.where(labels == medoid)[0] # datapoints assigned to this medoid
            mask = np.ones(DM.shape)
            mask[np.ix_(cluster, cluster)] = 0 # unmask distances between points in this cluster
            masked_distances = np.ma.masked_array(DM, mask=mask, fill_value=np.inf)
            costs = masked_distances.sum(axis=1)
            new_medoids[self.medoids == medoid] = costs.argmin(axis=0, fill_value=np.inf)
        return new_medoids

    def _relabel(self):
        label_dict = {v: k for k, v in enumerate(self.medoids)}
        for i in range(len(self._labels)):
            self._labels[i] = label_dict[self._labels[i]]

    def fit(self, X,y):
        """
        Fit the KMedoids model to the given data.

        Parameters:
        - X (ndarray): The input data.

        Returns:
        - ndarray: The cluster labels assigned to each data point.

        """
        self._init(X)
        num_iter = 0
        while 1:
            new_labels = self._find_labels(X)
            # Check convergence after re-assignment
            if np.array_equal(new_labels, self._labels):
                self.num_iter = num_iter
                self._relabel()
                return self._labels

            new_medoids = self._find_medoids(new_labels, X)
            num_iter += 1
            self._labels = new_labels
            self.medoids = new_medoids

            if num_iter == self.max_T:
                self.num_iter = num_iter
                self._relabel()
                return self._labels

    @staticmethod
    def combine_centers(list_indx_centers, cluster_subseq, cluster_centers, dist_function):
        """
        Combine two centers given their sub-sequences (this is called from AutoK)
        """
        all_sub_to_add = []
        for lb in list_indx_centers:
            all_sub_to_add.extend(cluster_subseq[lb])
        DistMatrix = calcualte_Distance_Matrix(all_sub_to_add, dist_function)
        median_index = np.argmin(np.sum(np.array(DistMatrix), axis=1))
        median_vector = all_sub_to_add[median_index]
        return median_vector, all_sub_to_add
def my_complete_clustering(full_dist_between_clusters,cluster_centers,distance_function,merge_clust,verbose):
    new_points = cluster_centers
    all_dists = np.zeros((len(new_points), len(new_points)))

    for i in range(len(new_points)):
        for j in range(i, len(new_points)):
            if i == j:
                continue
            if len(new_points[j])==0 or len(new_points[i])==0:
                continue
            all_dists[i][j] = distance_function(new_points[i], new_points[j])
            all_dists[j][i] = all_dists[i][j]

    To_merge = merge_clusters(full_dist_between_clusters,new_points, all_dists,merge_clust,verbose=verbose)

    return To_merge

def merge_clusters(full_dist_between_clusters,new_points, all_dists,merge_clust,verbose=False):
    clusters = [[i] for i in range(len(new_points))]
    merge_dists = []
    clusters_copy = []
    for i in range(len(new_points) - 1):
        clusters, merge_distance = find_positions_of_merge(full_dist_between_clusters,clusters, all_dists)
        clusters_copy.append(clusters.copy())
        merge_dists.append(merge_distance)
    if len(merge_dists) > 1:
        limit = statistics.median(merge_dists) + merge_clust * statistics.stdev(merge_dists)
        old_limit=0
        temp_dists=[ddd for ddd in merge_dists]
        while old_limit!=limit:
            temp_dists=[ddd for ddd in temp_dists if ddd<limit]
            old_limit=limit
            limit=statistics.mean(temp_dists) + merge_clust * statistics.stdev(temp_dists)
            if verbose:
                print(f"    {temp_dists}")
                print(f"    merge mean: {statistics.mean(temp_dists) }")
                print(f"    merge stdev: {statistics.stdev(temp_dists) }")
                print(f"    merge std*factor: {merge_clust *  statistics.stdev(temp_dists) }")
                print(f"    limit mean {statistics.mean(temp_dists) + merge_clust * statistics.stdev(temp_dists)}")
                print(f"    -------------------")
        for i, md in enumerate(merge_dists):
            if limit < md:
                if verbose:
                    print(clusters_copy[i])
                return clusters_copy[i - 1]
        if verbose:
            print(clusters_copy[-1])
    return clusters_copy[-1]


def find_positions_of_merge(full_dist_between_clusters,clusters_list, all_dist):
    mind = np.inf
    minpos = (0, 0)
    for i in range(len(clusters_list)):
        for j in range(i + 1, len(clusters_list)):
            if i == j:
                continue
            full_dist = full_dist_between_clusters(all_dist, clusters_list[i], clusters_list[j])
            if full_dist < mind:
                mind = full_dist
                minpos = (i, j)
    merge_distance = mind
    if minpos[0] == minpos[1]:
        return clusters_list, merge_distance
    new_cluster = []
    new_cluster.extend(clusters_list[minpos[0]])
    new_cluster.extend(clusters_list[minpos[1]])

    new_cluster_list = [new_cluster]
    for i in range(len(clusters_list)):
        c = clusters_list[i]
        if i == minpos[0] or i == minpos[1]:
            continue
        new_cluster_list.append(c)

    return new_cluster_list, merge_distance

def full_min_dist_between_clusters(all_dist, cluster1_ind, cluster2_ind):
    maxd=np.inf
    # maxd = 0
    for ind1 in cluster1_ind:
        for ind2 in cluster2_ind:
            if all_dist[ind2][ind1] < maxd:
            #if all_dist[ind2][ind1] > maxd:
                maxd = all_dist[ind2][ind1]
    return maxd

def full_dist_between_clusters(all_dist, cluster1_ind, cluster2_ind):
    # maxd=np.inf
    maxd = 0
    for ind1 in cluster1_ind:
        for ind2 in cluster2_ind:
            # if all_dist[ind2][ind1] < maxd:
            if all_dist[ind2][ind1] > maxd:
                maxd = all_dist[ind2][ind1]
    return maxd


def checks_and_format(cluster_centers,all_subsequences,list_label):
    # cluster_centers = cluster_centers.reshape(-1, cluster_centers.shape[1])

    cluster_subseq = [[] for i in range(len(cluster_centers))]
    for lbl, subs in zip(list_label, all_subsequences):
        cluster_subseq[lbl].append(subs)

    non_empty_centers = []
    non_empty_cluster_subseq_s = []
    keeplabs = []
    for i, (cluster, cluster_subseq_s) in enumerate(zip(cluster_centers, cluster_subseq)):
        if len(cluster_subseq_s) > 0:
            keeplabs.append(i)
            non_empty_centers.append(cluster_centers[i])
            non_empty_cluster_subseq_s.append(cluster_subseq[i])
    cluster_centers = non_empty_centers
    cluster_subseq = non_empty_cluster_subseq_s

    return cluster_centers,cluster_subseq,keeplabs
def merge_clusters_(full_dist_between_clusters,combine_centers,list_label,medoids,all_subsequences,distance_function,merge_clust,verbose=False,):

    cluster_centers, cluster_subseq, keeplabs = checks_and_format(medoids, all_subsequences, list_label)

    tomerge = my_complete_clustering(full_dist_between_clusters,medoids, distance_function, merge_clust=merge_clust,verbose=verbose)

    final_list_label = [0 for i in range(len(all_subsequences))]
    counter_clus = 0

    final_cluster_centers = []
    final_cluster_subs = []
    for list_indx_centers in tomerge:

        for i in range(len(list_label)):
            match_to_old_labels = [keeplabs[lic] for lic in list_indx_centers]
            if list_label[i] in match_to_old_labels:
                final_list_label[i] = counter_clus
        counter_clus += 1

        ## This has to be implemented from baseline technqiue
        fc, fsubs = combine_centers(list_indx_centers, cluster_subseq, cluster_centers,
                                                  distance_function)
        final_cluster_centers.append(fc)
        final_cluster_subs.append(fsubs)

    return final_cluster_centers, final_list_label, final_cluster_subs

def combine_centers(list_indx_centers, cluster_subseq, cluster_centers, dist_function):
    """
    Combine two centers given their sub-sequences (this is called from AutoK)
    """
    all_sub_to_add = []
    for lb in list_indx_centers:
        all_sub_to_add.extend(cluster_subseq[lb])
    DistMatrix=calcualte_Distance_Matrix(all_sub_to_add,dist_function)
    median_index = np.argmin(np.sum(np.array(DistMatrix), axis=1))
    median_vector = all_sub_to_add[median_index]
    return median_vector, all_sub_to_add

def calcualte_Distance_Matrix(X,distance_function):
    DM = np.zeros((len(X), len(X)))
    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            if j < i:
                continue
            if i == j:
                DM[i, j] = 0
            dist = distance_function(x1, x2)
            DM[i, j] = dist
            DM[j, i] = dist
    return DM