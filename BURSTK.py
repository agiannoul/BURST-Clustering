import statistics
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
import numpy
from kshape.core import KShapeClusteringCPU

from tslearn.metrics import y_shifted_sbd_vec



class BURSTK_Clustering:
    def __init__(self,init_len,k,alpha):
        self.model =None
        self.init_len =init_len
        self.k = k
        self.alpha = alpha
    def fit(self,X,y):
        series_len=X.shape[1]
        self.model = BURSTK_inner(pattern_length=series_len, subsequence_length=series_len,
                          init_length=self.init_len * series_len,alpha=self.alpha,
                          batch_size=self.init_len * series_len, overlaping_rate=series_len,
                          merge_clust=-1,merge_existing=1,k=self.k,
                          to_plot=False,verbose=False)
        for row in X:  # -230]:
            res = self.model.feed_data([v for v in row])
            #print(res)
    def predict(self, X):
        predicts=[]
        for row in X:  # -230]:
            res = self.model.feed_data([v for v in row])
            clustername = self.model.calculate_real_time_cluster(np.array([v for v in row]))
            predicts.append(clustername)
        return predicts
    def get_parameters(self):
        return {
            "window":self.init_len,
            "k":self.k,
        }

class BURSTK_inner():

    def __init__(self, pattern_length, subsequence_length, k=20, alpha=0.3, init_length=None,
                 batch_size=None, overlaping_rate=10, merge_clust=4,
                 merge_existing=1,to_plot=True, verbose=True):

        self.to_plot = to_plot
        self.verbose = verbose
        self.merge_clust = merge_clust
        self.merge_existing=merge_existing
        # Configuration parameter
        self.current_time = 0
        self.mean = -1
        self.std = -1
        self.start = False
        # algorithm parameter
        self.k = k
        self.subsequence_length = subsequence_length
        self.pattern_length = pattern_length

        # real time evolving storage
        self.clusters = []
        self.cluster_names = []
        self.new_clusters_dist = []
        self.nm_current_weight = []
        self.S = []
        self.cluster_subseqs = []
        self.ts = []

        self.alpha = alpha
        self.init_length = init_length
        self.batch_size = batch_size
        self.overlaping_rate = overlaping_rate
        self.cluster_assign = {}
        self.cluster_count = 0


    def feed_data(self, X):
        to_return = {}
        to_return["clusters"] = []
        to_return["time series"] = []
        to_return["scores"] = []
        to_return["Batch_scores"] = []

        self.ts.extend(X)
        # print(len(self.ts))
        if len(self.ts) >= self.init_length and self.start == False:
            if self.verbose:
                print(f"{self.current_time} - {self.current_time + self.init_length}", end='-->')
                print("Intialize clusters")
            self._initialize()
            self._set_normal_model()

            self._run(self.ts[:min(len(self.ts), self.current_time)])  # update weights
            self.start = True

            self.keeplast = self.ts[-self.pattern_length - self.subsequence_length:]
            self.ts = self.ts[self.current_time:]
            self.cluster_count = 0
            return to_return
        elif len(self.ts) >= self.batch_size and self.start:
            if self.verbose:
                print(f"{self.current_time} - {self.current_time + self.batch_size}", end='-->')
                print("update clusters on last batch")

            self._run_next_batch()  # clauclate new clusters and merger
            self._set_normal_model()  # calculate weitghs of clusters

            self._run(self.ts[:self.batch_size])  # update old wights with new


            self.delete_clusters()  # delete clusters with small weight


            self.cluster_count = 0
            self.keeplast = self.ts[-self.pattern_length - self.subsequence_length:]
            self.ts = self.ts[self.batch_size:]

            return to_return

        return to_return

    def calculate_real_time_cluster(self, time_series):

        dist_to_all = []
        for clust_i in self.clusters:
            dist = self._sbd(self._zscore(time_series, ddof=1), clust_i[0])[0]
            dist_to_all.append(dist)
        belonged_cluster = dist_to_all.index(min(dist_to_all))

        return self.cluster_names[belonged_cluster]

    def _run(self, ts):
        all_join = []
        all_activated_weighted = []
        if len(self.nm_current_weight) != len(self.weights):
            self.nm_current_weight = self.nm_current_weight + self.weights[len(self.nm_current_weight):]

        for scores_sub_join, scores_sub_join_old, t_decay in zip(self.weights,
                                                                 self.nm_current_weight, self.time_decay):
            new_w = float(scores_sub_join) / float(1 + max(0, t_decay - self.batch_size))
            update_w = float(1 - self.alpha) * float(scores_sub_join_old) + float(self.alpha) * float(new_w)

            all_activated_weighted.append(update_w)


        self.nm_current_weight = all_activated_weighted

        return

    # MAIN METHODS:
    # - Initialization
    # - Theta update for next batch
    # - Score computaiton

    # Initialization of the model
    def _initialize(self):
        cluster_subseqs, clusters = self._kshape_subsequence(initialization=True)
        self.cluster_subseqs = cluster_subseqs
        all_mean_dist = []
        for i, (cluster, cluster_subseq) in enumerate(zip(clusters, cluster_subseqs)):
            self._set_initial_S(cluster_subseq, i, cluster[0])
            all_mean_dist.append(self._compute_mean_dist_subs(cluster[0], cluster_subseq))
        self.clusters = clusters
        self.new_clusters_dist = all_mean_dist
        self.current_time = self.init_length
        self.cluster_names = [i for i in range(len(clusters))]
        self.globalCount_cluster = len(clusters)

    # Model update for next batch
    def _run_next_batch(self):

        # Run K-Shape algorithm on the subsequences of the current batch
        cluster_subseqs, clusters = self._kshape_subsequence(initialization=False)

        # self.new_clusters_to_merge = clusters

        to_add = [[] for i in range(len(self.clusters))]
        new_c = []
        new_c_subs = []
        # Finding the clusters that match exisiting clusters
        # - Storing in to_add all the clusters that have to be merged with the existing clusters
        # - Storing in new_c tyhe new clusters to be added.
        plotax = None
        if self.to_plot:
            self.old_cluster = [kati for kati in self.clusters]
            self.old_cluster_subs = [kati for kati in self.cluster_subseqs]
            self.new_cluster_to_plot = [kati for kati in clusters]
            self.new_cluster_subs_to_plot = [kati for kati in cluster_subseqs]
            # plt.show()
        for ci, (cluster, cluster_subseq) in enumerate(zip(clusters, cluster_subseqs)):
            if len(cluster_subseq)<1:
                continue
            min_dist = np.inf
            tmp_index = -1
            for index_o, origin_cluster in enumerate(self.clusters):
                new_dist = self._sbd(origin_cluster[0], cluster[0])[0]
                if min_dist > new_dist:
                    min_dist = new_dist
                    tmp_index = index_o
            if tmp_index != -1:
                if self.verbose:
                    print(f"mindist: {min_dist} , new_clusters_dist {self.new_clusters_dist[tmp_index]} ({tmp_index})")

                if min_dist < self.merge_existing*self.new_clusters_dist[tmp_index]:
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
                new_centroid, _ = self._extract_shape_stream(all_sub_to_add, i, cur_c[0], initial=False)
                if len(all_index)==0:
                    ok='ok'
                new_clusters.append((self._clean_cluster_tslearn(new_centroid), all_index))
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
            self._set_initial_S(t_a[1], len(self.clusters) + i, t_a[0][0])
            new_clusters.append((t_a[0][0], t_a[0][1]))
            all_mean_dist.append(self._compute_mean_dist_subs(t_a[0][0], new_c_subs[i]))
            new_clusters_subs.append(new_c_subs[i])
            self.cluster_names.append(self.globalCount_cluster)
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

    def plot_clusters(self, todelete=[]):
        old_cluster = self.old_cluster
        old_cluster_subs = self.old_cluster_subs
        clusters = self.new_cluster_to_plot
        cluster_subseqs = self.new_cluster_subs_to_plot
        if self.to_plot:
            fig, ax = plt.subplots(max([len(clusters), len(old_cluster), len(self.clusters), 2]), 3)
            for axes, col in zip(ax[0], [f"Batch clusters {self.k}->{len(clusters)}", "Old Clusters", "After Merging"]):
                axes.set_title(col)
            for q, clust in enumerate(clusters):
                for series in cluster_subseqs[q]:
                    ax[q][0].plot(series, color="black", alpha=0.3)
                ax[q][0].plot(clust[0], color="red")
            for q, clust in enumerate(old_cluster):
                for series in old_cluster_subs[q]:
                    ax[q][1].plot(series, color="black", alpha=0.3)
                ax[q][1].plot(clust[0], color="red")
            for q, clust in enumerate(self.clusters):
                for series in self.cluster_subseqs[q]:
                    ax[q][2].plot(series, color="black", alpha=0.3)
                ax[q][2].plot(clust[0], color="red")
                if q in todelete:
                    ax[q][2].add_patch(plt.Rectangle((0, 0), len(clust[0]), max(clust[0]), edgecolor='magenta', lw=2))
            plt.show()

    # SBD distance
    def _sbd(self, x, y):
        ncc = self._ncc_c(x, y)
        idx = ncc.argmax()
        dist = 1 - ncc[idx]
        return dist, None

    # Core clustering computation unit
    def _kshape_subsequence(self, initialization=True):
        all_subsequences = []
        idxs = []

        if initialization:
            nb_subsequence = self.init_length
        else:
            nb_subsequence = self.batch_size

        for i in range(self.current_time,
                       min(self.current_time + nb_subsequence,
                           self.current_time + len(self.ts) - self.subsequence_length),
                       self.overlaping_rate):
            all_subsequences.append(self.ts[i - self.current_time:i + self.subsequence_length - self.current_time])
            idxs.append(i)

        cluster_centers_, list_label, cluster_subsequencis = self.apply_advance_kshape(all_subsequences, idxs)
        if self.verbose:
            print(f"final clusters = {len(cluster_centers_)}")
        # print(cluster_centers_)
        new_k = len(cluster_centers_)

        # new_k=self.k

        # ks = KShape(n_clusters=self.k, verbose=False)
        # list_label = ks.fit_predict(np.array(all_subsequences))
        # cluster_centers_=ks.cluster_centers_

        cluster_subseq = [[self._zscore(s, ddof=1) for s in series] for series in cluster_subsequencis]
        cluster_idx = [[] for i in range(new_k)]
        for lbl, idx in zip(list_label, idxs):
            cluster_idx[lbl].append(idx)

        # safety check
        new_cluster_subseq = []
        clusters = []

        for i in range(new_k):

            if len(cluster_subseq[i]) > 0 and len(cluster_idx)==0:
                ok='ok'
            if len(cluster_subseq[i]) > 0:
                new_cluster_subseq.append(cluster_subseq[i])
                # clusters.append((self._clean_cluster_tslearn(ks.cluster_centers_[i]), cluster_idx[i]))
                res=(self._clean_cluster_tslearn(cluster_centers_[i]), cluster_idx[i])
                clusters.append(res)
                if len(res[1])==0:
                    ok='ok'

        for kati in new_cluster_subseq:
            if len(kati)==0:
                ok='ok'
        return new_cluster_subseq, clusters

    def delete_clusters(self):
        weights = self.nm_current_weight
        if len(weights) < 2:
            if self.to_plot:
                self.plot_clusters([])
            return
        delete_w = []
        if self.verbose:
            print("========== Weights ==============")
            print(self.weights)
            print(self.nm_current_weight)
        for i, w in enumerate(self.nm_current_weight):
            if w < 0.1:  # meanw - self.delete_w * std:
                delete_w.append(i)

        if self.to_plot:
            self.plot_clusters(sorted(delete_w, reverse=True))

        for ind in sorted(delete_w, reverse=True):
            del self.S[ind]
            del self.clusters[ind]
            del self.weights[ind]
            self.cluster_names.remove(self.cluster_names[ind])
            del self.nm_current_weight[ind]

            del self.cluster_subseqs[ind]

        if self.verbose:
            print(self.nm_current_weight)

            print("=================================")

    # Model elements update
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
                    dist_nms += self._sbd(nm[0], nm_t[0])[0]
            Centrality.append(dist_nms)

        Frequency = list((np.array(Frequency) - min(Frequency)) / (max(Frequency) - min(Frequency) + 0.0000001) + 1)
        Centrality = list(
            (np.array(Centrality) - min(Centrality)) / (max(Centrality) - min(Centrality) + 0.0000001) + 1)

        weights = []
        for f, c, t in zip(Frequency, Centrality, Time_decay):
            weights.append(float(f) ** 2 / float(c))

        self.weights = weights

        self.time_decay = Time_decay

    # Setting in memory the matrix S
    def _set_initial_S(self, X, idx, cluster_centers):
        # X = to_time_series_dataset(X)
        X=np.array(X)
        X = np.expand_dims(X, axis=-1)
        #cluster_centers = to_time_series(cluster_centers)
        cluster_centers = np.array(cluster_centers)
        cluster_centers = np.expand_dims(cluster_centers, axis=-1)
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(cluster_centers, X,
                               norm_ref=-1,
                               norms_dataset=np.linalg.norm(X, axis=(1, 2)))
        S = np.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        self.S.append(S)

    # Computation of the updated centroid
    def _extract_shape_stream(self, X, idx, cluster_centers, initial=True):
        X = np.array(X)
        X = np.expand_dims(X, axis=-1)
        cluster_centers = np.array(cluster_centers)
        cluster_centers = np.expand_dims(cluster_centers, axis=-1)
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(cluster_centers, X,
                               norm_ref=-1,
                               norms_dataset=np.linalg.norm(X, axis=(1, 2)))
        S = np.dot(Xp[:, :, 0].T, Xp[:, :, 0])

        if not initial:
            S = S + self.S[idx]
        self.S[idx] = S
        Q = np.eye(sz) - np.ones((sz, sz)) / sz
        M = np.dot(Q.T, np.dot(S, Q))
        _, vec = np.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))
        dist_plus_mu = np.sum(np.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = np.sum(np.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return self._zscore(mu_k, ddof=1), S

    # Reset value of a cluster
    def _clean_cluster_tslearn(self, cluster):
        return np.array([val[0] for val in cluster])

    def _compute_mean_dist_subs(self, cluster, subs):
        dist_all = []
        for sub in subs:
            dist_all.append(self._sbd(sub, cluster)[0])
        return np.mean(dist_all)

    def _running_mean(self, x, N):
        if len(x) < N:
            return np.array([np.mean(x)])
        smoothed_data = []
        for i in range(len(x) - N + 1):
            window = x[i:i + N]
            window_mean = np.mean(window)
            smoothed_data.append(window_mean)

        return np.array(smoothed_data)

    def _ncc_c(self, x, y):
        den = np.array(norm(x) * norm(y))
        den[den == 0] = np.inf

        x_len = len(x)
        fft_size = 1 << (2 * x_len - 1).bit_length()
        cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
        cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
        return np.real(cc) / den

    def _zscore(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        mns = a.mean(axis=axis)
        sstd = a.std(axis=axis, ddof=ddof)
        if axis and mns.ndim < a.ndim:
            res = ((a - np.expand_dims(mns, axis=axis)) /
                   np.expand_dims(sstd, axis=axis))
        else:
            if sstd>0:
                res = (a - mns) / sstd
            else:
                res = (a - mns)
        return np.nan_to_num(res)

    def apply_advance_kshape(self, all_subsequences, idxs):

        ks = KShapeClusteringCPU(self.k, centroid_init='random', max_iter=100, n_jobs=1)
        X = np.array(all_subsequences)
        X=X.reshape(X.shape[0], X.shape[1], 1)
        ks.fit(X)
        list_label = ks.predict(X)
        cluster_centers=ks.centroids_
        cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], cluster_centers.shape[1])


        # # old Kshape
        # ks = KShape(n_clusters=self.k, verbose=False)
        # list_label = ks.fit_predict(np.array(all_subsequences))
        #cluster_centers = ks.cluster_centers_


        cluster_centers = cluster_centers.reshape(-1, cluster_centers.shape[1])
        final_cluster_centers = []
        cluster_subseq = [[] for i in range(self.k)]
        for lbl, idx in zip(list_label, idxs):
            cluster_subseq[lbl].append(
                self.ts[idx - self.current_time:idx - self.current_time + self.subsequence_length])

        non_empty_centers = []
        non_empty_cluster_subseq_s = []
        keeplabs=[]
        for i, (cluster, cluster_subseq_s) in enumerate(zip(cluster_centers, cluster_subseq)):
            if len(cluster_subseq_s) > 0:
                keeplabs.append(i)
                non_empty_centers.append(cluster_centers[i])
                non_empty_cluster_subseq_s.append(cluster_subseq[i])
        cluster_centers = non_empty_centers
        cluster_subseq = non_empty_cluster_subseq_s

        if self.merge_clust<=0:
            tomerge = [[i] for i in range(len(cluster_centers))]
        else:
            tomerge = self.my_complete_clustering(cluster_centers)

        # self._extract_shape_stream(all_sub_to_add, i, cur_c[0], initial=False)
        final_cluster_subs = []
        final_list_label = [0 for i in range(len(all_subsequences))]
        counter_clus = 0
        for list_indx_centers in tomerge:

            for i in range(len(list_label)):
                match_to_old_labels=[keeplabs[lic] for lic in list_indx_centers]
                if list_label[i] in match_to_old_labels:
                    final_list_label[i] = counter_clus
            counter_clus += 1

            if len(list_indx_centers) > 1:
                all_sub_to_add = []
                for lb in list_indx_centers:
                    all_sub_to_add.extend(cluster_subseq[lb])
                if len(all_sub_to_add)>0:
                    new_center = self._shape_extraction(np.array(all_sub_to_add), cluster_centers[list_indx_centers[0]])
                    final_cluster_centers.append(new_center)

                    final_cluster_subs.append(all_sub_to_add)
                    if self.verbose:
                        print("merged")


            else:
                cnt = self._zscore(
                    np.array(cluster_centers[list_indx_centers[0]]).reshape(len(cluster_centers[list_indx_centers[0]]),
                                                                            1), ddof=1)
                final_cluster_centers.append(cnt)
                final_cluster_subs.append(cluster_subseq[list_indx_centers[0]])

        # final_cluster_centers=np.array(final_cluster_centers)
        # final_cluster_centers=final_cluster_centers.reshape(final_cluster_centers.shape[0],final_cluster_centers.shape[1],1)

        f_clusters=[]
        return np.array(final_cluster_centers), final_list_label, final_cluster_subs

    def _shape_extraction(self, Xs, forshift):
        sz = Xs.shape[1]
        tempforshift = np.array(forshift)
        tempforshift = tempforshift.reshape(tempforshift.shape[0], 1)
        X = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)

        Xp = y_shifted_sbd_vec(tempforshift, X,
                               norm_ref=-1,
                               norms_dataset=np.linalg.norm(X, axis=(1, 2)))

        S = numpy.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        Q = numpy.eye(sz) - numpy.ones((sz, sz)) / sz
        M = numpy.dot(Q.T, numpy.dot(S, Q))
        _, vec = numpy.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))

        # The way the optimization problem is (ill-)formulated, both mu_k and
        # -mu_k are candidates for barycenters
        # In the following, we check which one is best candidate
        dist_plus_mu = numpy.sum(numpy.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = numpy.sum(numpy.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return self._zscore(mu_k, ddof=1)

    def my_complete_clustering(self, cluster_centers):
        new_points = cluster_centers
        all_dists = np.zeros((len(new_points), len(new_points)))

        for i in range(len(new_points)):
            for j in range(i, len(new_points)):
                if i == j:
                    continue
                if len(new_points[j])==0 or len(new_points[i])==0:
                    continue
                all_dists[i][j] = self._sbd(new_points[i], new_points[j])[0]
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