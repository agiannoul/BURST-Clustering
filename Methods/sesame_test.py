from river import cluster
from river import stream

from Methods.distances import get_transformation
from pysame import Birch


def make_model(class_ref,transformation,**params):
    return class_ref(transformation=transformation,**params)


class Clustream:
    def __init__(self,k,slide,time_window_factor,maxfactor,halflife,distance_measure="euclidean",transformation="default"):
        self.k = k
        self.distance_measure = transformation
        self.transformation_name = transformation
        self.transformation = get_transformation(distance_measure,transformation)
        self.slide = slide
        self.halflife = halflife
        self.time_window_factor = time_window_factor
        self.model= cluster.CluStream(time_window=self.slide,
        max_micro_clusters = self.k*maxfactor,
        n_macro_clusters = self.k,
        time_gap = self.slide,
        seed = 0,
        halflife = self.halflife)

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