import numpy as np
from river import cluster
from river import stream
from sklearn.cluster import MiniBatchKMeans as MBKMeans
from Methods.distances import get_transformation



def make_model(class_ref,transformation,**params):
    return class_ref(transformation=transformation,**params)


import numpy as np
from sklearn.cluster import Birch



class BirchN:
    def __init__(self,slide,distance_measure="euclidean",transformation="default",threshold=0.5):
        self.distance_measure = transformation
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure,transformation)
        self.slide = slide
        self.model=Birch(n_clusters=None,threshold=threshold)

    def fit(self, X,y):
        X = self.transformation(X)
        self.model.fit(X)

    def predict(self,X):
        X = self.transformation(X)
        predicts = []
        for i, x in enumerate(X):
            predicts.append(self.model.predict(np.array([x]))[0])
            if i>self.slide and i%self.slide ==0:
                lastbatch=X[i-self.slide:i]
                self.model.partial_fit(lastbatch)
        return predicts


    def get_parameters(self):
        return {
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }

class BirchK:
    def __init__(self,slide=100,k=3,threshold=0.5,distance_measure="euclidean",transformation="default"):
        self.distance_measure = transformation
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure,transformation)
        self.slide = slide
        self.k = k
        self.model=Birch(n_clusters=k,threshold=threshold)

    def fit(self, X,y):
        X = self.transformation(X)
        self.model.fit(X)

    def predict(self,X):
        X = self.transformation(X)
        predicts = []
        for i, x in enumerate(X):
            predicts.append(self.model.predict(np.array([x]))[0])
            if i>self.slide and i%self.slide ==0:
                lastbatch=X[i-self.slide:i]
                self.model.partial_fit(lastbatch)
        return predicts


    def get_parameters(self):
        return {
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
            "k":self.k,
        }


class Clustream:
    def __init__(self,k,slide,maxfactor,halflife=0.5,n_clusters=None,distance_measure="euclidean",transformation="default"):
        self.k = k
        self.distance_measure = transformation
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure,transformation)
        self.slide = slide
        self.halflife = halflife
        if n_clusters is None:
            self.model= cluster.CluStream(time_window=self.slide,
            max_micro_clusters = maxfactor,
            n_macro_clusters = self.k,
            time_gap = self.slide,
            seed = 0,
            halflife = self.halflife)
        else:
            self.model = cluster.CluStream(time_window=self.slide,
                           max_micro_clusters=maxfactor,
                           n_macro_clusters=self.k,
                           time_gap=self.slide,
                           seed=0,
                           halflife=self.halflife,
                           n_clusters = n_clusters
                           )

    def fit(self, X,y):
        X = self.transformation(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
    def predict(self,X):
        X = self.transformation(X)
        predicts=[]
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
            predicts.append(self.model.predict_one(x))
        return predicts
    def get_parameters(self):
        return {
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
            "k":self.k,
        }
from clusopt_core.cluster import CluStream

class Clustream2:
    def __init__(self,k,slide,radiusfactor=1,k_multiplier=2,distance_measure="euclidean",transformation="default"):
        self.k = k
        self.distance_measure = transformation
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure,transformation)
        self.slide = slide
        self.model=CluStream(
            m=min(k * k_multiplier,slide),  # no microclusters
            h=slide,  # horizon
            t=radiusfactor,  # radius factor
        )

    def fit(self, X,y):
        X = self.transformation(X)
        # for i, (x, _) in enumerate(stream.iter_array(X)):
        self.model.init_offline(X)
        self.clusters, _ = self.model.get_macro_clusters(self.k, seed=42)

    def predict(self,X):
        X = self.transformation(X)
        predicts=[]
        batch=[]
        for x in X:
            predicts.append(self.predict_one(x))
            batch.append(x)
            if len(batch)>=self.slide:
                self.model.partial_fit(np.array(batch))
                self.clusters, _ = self.model.get_macro_clusters(self.k, seed=42)
                batch=[]
        return predicts
    def predict_one(self,x):
        distances = np.linalg.norm(self.clusters - x, axis=1)
        return np.argmin(distances)
    def get_parameters(self):
        return {
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
            "k":self.k,
        }


class DbStream:
    def __init__(self,slide,fading_factor,cleanup_interval,intersection_factor,clustering_threshold,minimum_weight=1,distance_measure="euclidean",transformation="default",k=None):
        self.transformation_name=transformation
        self.distance_measure=distance_measure
        self.transformation = get_transformation(distance_measure, transformation)

        self.fading_factor=fading_factor
        self.slide=slide
        self.cleanup_interval=cleanup_interval
        self.intersection_factor=intersection_factor
        self.model=cluster.DBSTREAM(clustering_threshold = clustering_threshold,
                            fading_factor = self.fading_factor,
                            cleanup_interval = self.slide,
                            intersection_factor = cleanup_interval,
                            minimum_weight = minimum_weight)
    def fit(self, X,y):
        X = self.transformation(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
    def predict(self,X):
        X = self.transformation(X)
        predicts=[]
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
            predicts.append(self.model.predict_one(x))
        return predicts
    def get_parameters(self):
        return {
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }


class DenStream:
    def __init__(self,slide,epsilon,mu_factor,beta,decaying_factor,distance_measure="euclidean",transformation="default",k=None):
        self.transformation = get_transformation(distance_measure, transformation)
        self.transformation_name = transformation
        self.distance_measure = distance_measure
        self.init_size = slide
        self.beta = beta
        self.model=cluster.DenStream(decaying_factor=decaying_factor,
                              beta=beta,
                               mu=mu_factor,
                              epsilon=epsilon,
                              n_samples_init=slide,stream_speed=1)
    def fit(self, X,y):
        X = self.transformation(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
    def predict(self,X):
        X = self.transformation(X)
        predicts=[]
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
            predicts.append(self.model.predict_one(x))
        return predicts
    def get_parameters(self):
        return {
            "init_size":self.init_size,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
        }


class MiniBatchKMeans:
    def __init__(self,k,slide,distance_measure="euclidean",transformation="default"):
        self.transformation = get_transformation(distance_measure, transformation)
        self.k=k
        self.distance_measure=distance_measure
        self.transformation_name=transformation
        self.slide=slide
        self.model = MBKMeans(n_clusters=k,
                                 random_state=0,
                                 batch_size=slide,
                                 n_init="auto")
    def fit(self, X, y):
        X = self.transformation(X)
        # for i, (x, _) in enumerate(stream.iter_array(X)):
        self.model.partial_fit(np.array(X))
        self.clusters= self.model.cluster_centers_

    def predict(self, X):
        X = self.transformation(X)
        predicts = []
        batch = []
        for x in X:
            predicts.append(self.predict_one(x))
            batch.append(x)
            if len(batch) >= self.slide:
                self.model.partial_fit(np.array(batch))
                self.clusters= self.model.cluster_centers_
                batch = []
        return predicts

    def predict_one(self, x):
        distances = np.linalg.norm(self.clusters - x, axis=1)
        return np.argmin(distances)
    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
            "slide":self.slide
        }


class STREAMLS:
    def __init__(self,k,slide,halflife=1,distance_measure="euclidean",transformation="default"):
        self.transformation = get_transformation(distance_measure, transformation)
        self.k=k
        self.distance_measure=distance_measure
        self.transformation_name=transformation
        self.slide=slide
        self.model=cluster.STREAMKMeans(chunk_size=self.slide,
                                        n_clusters=self.k,
                                        halflife=halflife,
                                        seed=0)
    def fit(self, X,y):
        X = self.transformation(X)
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
    def predict(self,X):
        X = self.transformation(X)
        predicts=[]
        for i, (x, _) in enumerate(stream.iter_array(X)):
            self.model.learn_one(x)
            predicts.append(self.model.predict_one(x))
        return predicts
    def get_parameters(self):
        return {
            "k":self.k,
            "distance_measure":self.distance_measure,
            "transformation":self.transformation_name,
            "slide":self.slide
        }