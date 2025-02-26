import abc

import numpy as np
from sklearn.cluster import KMeans as KMeanscore
from .distances import  get_distance_measure
from .distances import  get_transformation
random_state=42

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
class BatchModel:
    @abc.abstractmethod
    def fit(self, X, y):
        """

        :param X: distances
        :param y:
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """

        :param X: distances
        :return:
        """
        pass

    def tuned_fit(self, X,y,metric):
        """

        :param X:
        :param y:
        :param metric:
        :return:
        """
        pass


class Kmeans(BatchModel):
    def __init__(self,k=2,distance_measure='euclidean',transformation="default"):
        self.k=k
        self.transformation_name = transformation
        self.distance_measure=distance_measure
        self.transformation=get_transformation(distance_measure=distance_measure,transformation=transformation)
        if distance_measure != 'euclidean':
            raise ValueError('kmeans only supports euclidean distance')
        self.model=KMeanscore(n_clusters=self.k, random_state=random_state, n_init="auto")
    def fit(self, X, y):
        X=self.transformation(X)
        self.model = self.model.fit(X)
        self.cluster_centers_=self.model.cluster_centers_

    def predict(self, X):
        X = self.transformation(X)
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


from .kmediods import KMedoids

class Kmedoids(BatchModel):
    def __init__(self,k=2,distance_measure='euclidean',transformation="default"):#catch22
        self.k=k
        self.transformation_name=transformation
        self.transformation=get_transformation(distance_measure=distance_measure,transformation=transformation)
        self.distance_measure=distance_measure
        self.model=None
        self.distance_function=get_distance_measure(distance_measure)

    def fit(self, X, y):
        X = self.transformation(X)
        DM=calcualte_Distance_Matrix(X, self.distance_function)
        self.model = KMedoids(self.k)
        self.model.fit(DM,y)
        self.medoids=[]
        for i in self.model.medoids:
            self.medoids.append(X[i])
        self.cluster_centers_=np.array(self.medoids)
    def predict(self, X):
        X = self.transformation(X)
        labels=[]
        for x1 in X:
            distances=[]
            for med in self.medoids:
                distances.append(self.distance_function(x1,med))
            lab=np.argmin(distances)
            labels.append(lab)
        return labels

    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }

    @staticmethod
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

    @staticmethod
    def meta_store(meta_data, cluster_subseq, cluster_center, name,dist_function):
        """
        Store meta date, for kmedoids we store the number of subsequences in each cluster (count) and the mean distance from the center
        """
        if name not in meta_data:
            meta_data[name] = {}
        meta_data[name]['count'] = len(cluster_subseq)
        dist_from_center=sum([dist_function(cluster_center,clu_s) for clu_s in cluster_subseq])/len(cluster_subseq)
        meta_data[name]['dist_from_center'] = dist_from_center
        return meta_data

    @staticmethod
    def combine_old_new(meta_data, name, new_subs, old_center, new_centers, dist_function):
        """
        Combine old clusters (from previous batches with new clusters from current batch)
        """
        if len(new_centers)>1:
            DistMatrix = calcualte_Distance_Matrix(new_subs, dist_function)
            median_index = np.argmin(np.sum(np.array(DistMatrix), axis=1))
            new_center = new_subs[median_index]
        else:
            new_center=new_centers[0]
        if meta_data[name]['dist_from_center']!=0:
            old_w=meta_data[name]['count']/meta_data[name]['dist_from_center']
            cnew=len(new_subs)
            if cnew<=1 or meta_data[name]['count']<=1:
                if meta_data[name]['count']>cnew:
                    return old_center, meta_data
                else:
                    meta_data[name]["count"] = cnew
                    meta_data[name]["dist_from_center"] = 0
                    return new_center, meta_data
            distnew=sum([dist_function(new_center,clu_s) for clu_s in new_subs])/len(new_subs)
            new_w=cnew/distnew
            if old_w>new_w:
                return old_center,meta_data
            else:
                meta_data[name]["count"]=cnew
                meta_data[name]["dist_from_center"]=distnew
                return new_center,meta_data
        else:
            cnew = len(new_subs)
            distnew = sum([dist_function(new_center, clu_s) for clu_s in new_subs]) / len(new_subs)
            meta_data[name]["count"] = cnew
            meta_data[name]["dist_from_center"] = distnew
            return new_center, meta_data


from kshape.core import KShapeClusteringCPU
class KShapes(BatchModel):
    def __init__(self,k=2,distance_measure='euclidean',transformation="default"):
        self.k=k
        if transformation!='default':
            raise ValueError('kshapes only supports default shape')
        self.transformation_name=transformation
        self.distance_measure=distance_measure
        if distance_measure != 'SBD':
            raise ValueError('KShape only supports SBD distance')
        self.model=None
    def fit(self, X, y):

        newX=[]
        for x in X:
            newX.append(_zscore(x, ddof=1))
        X=np.array(newX)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model = KShapeClusteringCPU(self.k, centroid_init='random', max_iter=100, n_jobs=1)
        self.model.fit(X)

    def predict(self, X):
        newX = []
        for x in X:
            newX.append(_zscore(x, ddof=1))
        X = np.array(newX)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X)
    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }

from .DTC.main import dtc
class DTCluster(BatchModel):
    def __init__(self,k=2,distance_measure='euclidean',transformation="default"):
        self.k=k
        self.X=None
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure=distance_measure, transformation=transformation)
        self.distance_measure = distance_measure
    def fit(self, X, y):
        self.k=min(self.k,len(X))
        X = self.transformation(X)
        predictions,self.model  = dtc(X, y, self.k)

    def predict(self, X):
        X = self.transformation(X)
        x = np.expand_dims(X, axis=2)
        if x.shape[1] % 2 != 0:
            npad = ((0, 0), (0, 1), (0, 0))
            x = np.pad(x, npad, mode='constant', constant_values=0)
            print(x.shape)
        return self.model.predict(x)[1].argmax(axis=1)

    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }

def _zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)
