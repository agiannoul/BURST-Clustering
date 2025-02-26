import statistics

import numpy as np

from Methods.distances import get_distance_measure, get_transformation

DEBUG=False

class BURST_GEN:
    def __init__(self,baseline,slide,k=None,alpha=0.3,distance_measure='euclidean',transformation='default',verbose=False):
        self.baseline = baseline
        self.alpha=alpha
        self.distance_measure = distance_measure
        self.batch_size = slide
        self.k = k
        self.transformation_name = transformation


        self.transformation = get_transformation(distance_measure=distance_measure, transformation=transformation)

        self.big_k=20
        self.merge_clust=2
        self.distance_function = get_distance_measure(distance_measure)

        self.verbose=verbose

        self.batch=[]
        self.current_time=0
        self.meta_data={}
        self.cluster_names = []
        self.nm_current_weight=[]
        self.initialize=False


    def delete_clusters(self):
        weights = self.nm_current_weight
        delete_w = []
        if self.verbose:
            print("========== Weights ==============")
            print(self.weights)
            print(self.nm_current_weight)
        for i, w in enumerate(self.nm_current_weight):
            if w < 0.1:  # meanw - self.delete_w * std:
                delete_w.append(i)
        OK='OK'
        for ind in sorted(delete_w, reverse=True):
            del self.meta_data[self.cluster_names[ind]]
            del self.clusters[ind]
            del self.weights[ind]
            self.cluster_names.remove(self.cluster_names[ind])
            del self.nm_current_weight[ind]

            del self.cluster_subseqs[ind]

        if self.verbose:
            print(self.nm_current_weight)

            print("=================================")



    def fit(self,X,y):
        X = self.transformation(X)
        self.feed_data(X)
    def predict(self,X):
        X = self.transformation(X)
        if self.initialize:
            return self.feed_data(X)
        else:
            raise ValueError("Clusters are not initialized")


    def feed_data(self,X):
        self.predicts=[]
        for x in X:
            self.batch.append(x)
            if self.initialize:
                self.predicts.append(self.calculate_real_time_cluster(x))
            if len(self.batch)==self.batch_size:
                if self.initialize==False:
                    self._initialize()
                    self.initialize=True
                else:
                    self._run_next_batch()  # clauclate new clusters and merger
                self._set_normal_model()  # calculate weitghs of clusters
                self._run()  # update old wights with new
                self.delete_clusters()
                self.batch=[]
        return self.predicts
    def calculate_real_time_cluster(self, time_series):

        dist_to_all = []
        for clust_i in self.clusters:
            dist = self.distance_function(time_series,clust_i[0])
            dist_to_all.append(dist)
        belonged_cluster = dist_to_all.index(min(dist_to_all))
        return self.cluster_names[belonged_cluster]

    def _run(self):
        all_activated_weighted = []
        if len(self.nm_current_weight) != len(self.weights):
            self.nm_current_weight = self.nm_current_weight + self.weights[len(self.nm_current_weight):]

        for NW, OLDw, t_decay in zip(self.weights,self.nm_current_weight, self.time_decay):
            new_w = float(NW) / float(1 + max(0, t_decay - self.batch_size))
            update_w = float(1 - self.alpha) * float(OLDw) + float(self.alpha) * float(new_w)

            all_activated_weighted.append(update_w)



        self.nm_current_weight = all_activated_weighted


        return  # list(np.nan_to_num(join))

    def _initialize(self):
        cluster_subseqs, clusters = self.core_clustering()
        self.cluster_subseqs = cluster_subseqs
        all_mean_dist = []
        self.cluster_names = [i for i in range(len(clusters))]
        for cluster, cluster_subseq,name in zip(clusters, cluster_subseqs,self.cluster_names):
            all_mean_dist.append(self._compute_mean_dist_subs(cluster[0], cluster_subseq))
            self.baseline.meta_store(self.meta_data, cluster_subseq, cluster[0], name,self.distance_function)
        self.clusters = clusters
        self.new_clusters_dist = all_mean_dist
        self.current_time = self.batch_size

        self.globalCount_cluster = len(clusters)


    def _set_normal_model(self):
        Frequency = []
        Centrality = []
        Time_decay = []
        for i, nm in enumerate(self.clusters):
            Frequency.append(float(len(nm[1])))
            if len(nm[1])==0:
                ok='ok'
            Time_decay.append(float(self.current_time) - float(max(nm[1])))
            dist_nms = 0
            for j, nm_t in enumerate(self.clusters):
                if j != i:
                    dist_nms += self.distance_function(nm[0], nm_t[0])
            Centrality.append(dist_nms)

        Frequency = list((np.array(Frequency) - min(Frequency)) / (max(Frequency) - min(Frequency) + 0.0000001) + 1)
        Centrality = list(
            (np.array(Centrality) - min(Centrality)) / (max(Centrality) - min(Centrality) + 0.0000001) + 1)

        weights = []
        for f, c, t in zip(Frequency, Centrality, Time_decay):
            weights.append(float(f) ** 2 / float(c))

        self.weights = weights

        self.time_decay = Time_decay

    # TO DO
    def _run_next_batch(self):

        # Run K-clust algorithm on the subsequences of the current batch
        cluster_subseqs, clusters = self.core_clustering()

        # self.new_clusters_to_merge = clusters

        to_add = [[] for i in range(len(self.clusters))]
        new_c = []
        new_c_subs = []
        # Finding the clusters that match existing clusters
        # - Storing in to_add all the clusters that have to be merged with the existing clusters
        # - Storing in new_c tyhe new clusters to be added.
        for ci, (cluster, cluster_subseq) in enumerate(zip(clusters, cluster_subseqs)):
            if len(cluster_subseq)<1:
                continue
            min_dist = np.inf
            tmp_index = -1
            for index_o, origin_cluster in enumerate(self.clusters):
                new_dist = self.distance_function(origin_cluster[0], cluster[0])
                if min_dist > new_dist:
                    min_dist = new_dist
                    tmp_index = index_o
            if tmp_index != -1:
                if self.verbose:
                    print(f"mindist: {min_dist} , new_clusters_dist {self.new_clusters_dist[tmp_index]} ({tmp_index})")

                if min_dist < self.new_clusters_dist[tmp_index]:
                    to_add[tmp_index].append((cluster, cluster_subseq))
                    if self.verbose:
                        print(f"Cluster {ci} is merged with old cluster {tmp_index}")
                else:
                    if self.verbose:
                        print(f"Cluster new added {ci}")

                    new_c.append((cluster, cluster_subseq))
                    new_c_subs.append(cluster_subseqs[ci])
        self.to_add = to_add
        self.new_c = new_c

        new_clusters = []
        all_mean_dist = []
        new_clusters_subs = []
        # Merging existing clusters with new clusters
        for i, (cur_c, t_a) in enumerate(zip(self.clusters, to_add)):
            # Check if new subsequences to add
            if len(t_a) > 0:
                all_index = cur_c[1]
                all_sub_to_add = []
                for t_a_s in t_a:
                    all_index += t_a_s[0][1]
                    all_sub_to_add += t_a_s[1]

                # Updating the centroid shape
                #new_centroid, _ = self._extract_shape_stream(all_sub_to_add, i, cur_c[0], initial=False)
                new_centroid,self.meta_data=self.baseline.combine_old_new(self.meta_data,self.cluster_names[i],all_sub_to_add,cur_c[0],t_a[0][0],self.distance_function)

                if len(all_index)==0:
                    ok='ok'
                #new_clusters.append((self._clean_cluster_tslearn(new_centroid), all_index))
                new_clusters.append((np.array(new_centroid.ravel()), all_index))
                new_clusters_subs.append(all_sub_to_add)
                # Updating the intra cluster distance
                dist_to_add = self._compute_mean_dist_subs(cur_c[0], all_sub_to_add)
                ratio = float(len(cur_c[1])) / float(len(cur_c[1]) + len(all_index))
                all_mean_dist.append((ratio) * self.new_clusters_dist[i] + (1.0 - ratio) * dist_to_add)

            # If no new subsequences to add, copy the old cluster
            else:
                if len(cur_c[1])==0:
                    ok='ok'
                new_clusters.append(cur_c)
                all_mean_dist.append(self.new_clusters_dist[i])
                new_clusters_subs.append(self.cluster_subseqs[i])

        # Adding new clusters
        for i, t_a in enumerate(new_c):
            self.cluster_names.append(self.globalCount_cluster)
            self.baseline.meta_store(self.meta_data,t_a[1], t_a[0][0],self.cluster_names[-1],self.distance_function)
            new_clusters.append((t_a[0][0], t_a[0][1]))
            all_mean_dist.append(self._compute_mean_dist_subs(t_a[0][0], new_c_subs[i]))
            new_clusters_subs.append(new_c_subs[i])

            self.globalCount_cluster += 1


        self.clusters = new_clusters
        for clust in self.clusters:
            if len(clust[1])==0:
                ok='ok'
        self.cluster_subseqs = new_clusters_subs
        if self.verbose:
            print(f"FINAL: {len(self.clusters)}")
        self.new_clusters_dist = all_mean_dist
        self.current_time = self.current_time + self.batch_size


    def core_clustering(self,):
        idxs = []

        all_subsequences=self.batch
        idxs.extend([self.current_time+i for i in range(len(self.batch))])

        cluster_centers_, list_label, cluster_subsequencis = self.batch_clustering(all_subsequences, idxs)
        if self.verbose:
            print(f"final clusters = {len(cluster_centers_)}")
        # print(cluster_centers_)

        new_k = len(cluster_centers_)

        cluster_subseq = [[s for s in series] for series in cluster_subsequencis]
        cluster_idx = [[] for i in range(new_k)]
        for lbl, idx in zip(list_label, idxs):
            cluster_idx[lbl].append(idx)

        # safety check
        new_cluster_subseq = []
        clusters = []

        for i in range(new_k):
            if len(cluster_subseq[i]) > 0:
                new_cluster_subseq.append(cluster_subseq[i])
                res = (np.array(cluster_centers_[i].ravel()), cluster_idx[i])
                clusters.append(res)
        return new_cluster_subseq, clusters


    def batch_clustering(self,all_subsequences, idxs):

        if self.k is not None:
            tempk = min(self.k, len(all_subsequences))
        else:
            tempk=min(self.big_k,len(all_subsequences))
        if self.distance_measure!="euclidean":
            ks = self.baseline(k=tempk,distance_measure=self.distance_measure)
        else:
            ks = self.baseline(k=tempk)
        ks.fit(np.array(all_subsequences),None)
        list_label = ks.predict(np.array(all_subsequences))
        list_label=[lb for lb in list_label]
        cluster_centers=ks.cluster_centers_
        if DEBUG:
            plot_clusters_with_pca(all_subsequences,list_label)
        cluster_centers,cluster_subseq,keeplabs=self.checks_and_format( cluster_centers, all_subsequences, list_label)

        if self.k is not None:
            tomerge=[[i] for i in range(len(cluster_centers))]
        else:
            tomerge = self.my_complete_clustering(cluster_centers)

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
            fc,fsubs=self.baseline.combine_centers(list_indx_centers,cluster_subseq,cluster_centers,self.distance_function)
            final_cluster_centers.append(fc)
            final_cluster_subs.append(fsubs)
        if DEBUG:
            toplotsubs=[]
            toplot_labels=[]
            for i,clustsubs in enumerate(final_cluster_subs):
                toplotsubs.extend([sub for sub in clustsubs])
                toplot_labels.extend([i for q in clustsubs])
            plot_clusters_with_pca(toplotsubs,toplot_labels)
        return np.array(final_cluster_centers), final_list_label, final_cluster_subs

    # def combine_centers(self,list_indx_centers,cluster_subseq,cluster_centers):
    #     final_cluster_centers = []
    #     final_cluster_subs = []
    #     if len(list_indx_centers) > 1:
    #         all_sub_to_add = []
    #         for lb in list_indx_centers:
    #             all_sub_to_add.extend(cluster_subseq[lb])
    #         if len(all_sub_to_add) > 0:
    #             new_center = self._shape_extraction(np.array(all_sub_to_add), cluster_centers[list_indx_centers[0]])
    #             final_cluster_centers.append(new_center)
    #
    #             final_cluster_subs.append(all_sub_to_add)
    #             if self.verbose:
    #                 print("merged")
    #
    #
    #     else:
    #         cnt = self._zscore(
    #             np.array(cluster_centers[list_indx_centers[0]]).reshape(len(cluster_centers[list_indx_centers[0]]), 1),
    #             ddof=1)
    #         final_cluster_centers.append(cnt)
    #         final_cluster_subs.append(cluster_subseq[list_indx_centers[0]])
    def checks_and_format(self,cluster_centers,all_subsequences,list_label):
        cluster_centers = cluster_centers.reshape(-1, cluster_centers.shape[1])

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

    def my_complete_clustering(self, cluster_centers):
        new_points = cluster_centers
        all_dists = np.zeros((len(new_points), len(new_points)))

        for i in range(len(new_points)):
            for j in range(i, len(new_points)):
                if i == j:
                    continue
                if len(new_points[j])==0 or len(new_points[i])==0:
                    continue
                all_dists[i][j] = self.distance_function(new_points[i], new_points[j])
                all_dists[j][i] = all_dists[i][j]

        To_merge = self.merge_clusters(new_points, all_dists)

        return To_merge

    def merge_clusters(self, new_points, all_dists):
        clusters = [[i] for i in range(len(new_points))]
        merge_dists = []
        clusters_copy = []
        for i in range(len(new_points) - 1):
            clusters, merge_distance = self.find_positions_of_merge(clusters, all_dists)
            clusters_copy.append(clusters.copy())
            merge_dists.append(merge_distance)
        if len(merge_dists) > 2:

            limit = statistics.median(merge_dists) + self.merge_clust * statistics.stdev(merge_dists)
            old_limit=0
            temp_dists=[ddd for ddd in merge_dists]
            while old_limit!=limit:
                temp_dists=[ddd for ddd in temp_dists if ddd<limit]
                old_limit=limit
                limit=statistics.mean(temp_dists) + self.merge_clust * statistics.stdev(temp_dists)
                if self.verbose:
                    print(f"    {temp_dists}")
                    print(f"    merge mean: {statistics.mean(temp_dists) }")
                    print(f"    merge stdev: {statistics.stdev(temp_dists) }")
                    print(f"    merge std*factor: {self.merge_clust *  statistics.stdev(temp_dists) }")
                    print(f"    limit mean {statistics.mean(temp_dists) + self.merge_clust * statistics.stdev(temp_dists)}")
                    print(f"    -------------------")
            for i, md in enumerate(merge_dists):
                if limit < md:
                    if self.verbose:
                        print(clusters_copy[i])
                    return clusters_copy[i - 1]
            if self.verbose:
                print(clusters_copy[-1])
        return clusters_copy[-1]

    def full_dist_between_clusters(self, all_dist, cluster1_ind, cluster2_ind):
        # maxd=np.inf
        maxd = 0
        for ind1 in cluster1_ind:
            for ind2 in cluster2_ind:
                # if all_dist[ind2][ind1] < maxd:
                if all_dist[ind2][ind1] > maxd:
                    maxd = all_dist[ind2][ind1]
        return maxd

    def find_positions_of_merge(self, clusters_list, all_dist):
        mind = np.inf
        minpos = (0, 0)
        for i in range(len(clusters_list)):
            for j in range(i + 1, len(clusters_list)):
                if i == j:
                    continue
                full_dist = self.full_dist_between_clusters(all_dist, clusters_list[i], clusters_list[j])
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

    def _compute_mean_dist_subs(self, cluster, subs):
        dist_all = []
        for sub in subs:
            dist_all.append(self.distance_function(sub, cluster))
        return np.mean(dist_all)

    def get_parameters(self):

        return {
            "baseline": self.baseline.__name__,
            "distance_measure": self.distance_measure,
            "slide": self.batch_size,
            "transformation_name":self.transformation_name,
        }



def plot_clusters_with_pca(vectors, labels):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns

    """
    Plots clusters after applying PCA (2D projection).

    :param vectors: List or numpy array of shape (n_samples, n_features)
    :param labels: List or numpy array of shape (n_samples,) containing cluster labels
    """
    vectors = np.array(vectors)
    labels = np.array(labels)

    # Standardize the data
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    # Apply PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors_scaled)

    colors = sns.color_palette("Paired", len(set(labels)))

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        plt.scatter(
            reduced_vectors[labels == label, 0],
            reduced_vectors[labels == label, 1],
            label=f'Cluster {label}',
            # alpha=0.7,
            color=colors[i % 20]
        )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    # plt.legend()
    plt.show()