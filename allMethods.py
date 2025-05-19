import time
from math import sqrt

import numpy as np
from sklearn.metrics import rand_score, normalized_mutual_info_score, adjusted_rand_score

from BURST import BURST_Clustering
from BURSTK import BURSTK_Clustering
from BURST_Alternatives.BURSTK_NaiveMerge import BURSTK_Clustering_NaiveMerge
from BURST_Alternatives.BURST_2S import BURST_Clustering_2S
from BURST_GENERAL import BURST_GEN
from BURST_Alternatives.BURST_GENERAL_2S import BURST_GEN_2S
from BURST_Alternatives.BURST_GENERAL_NaiveMerge import BURST_GEN_NaiveMerge
from BURST_Alternatives.BURST_L import BURST_Clustering_L
from BURST_Alternatives.BURST_NaiveMerge import BURST_Clustering_NaiveMerge
from Methods import trivialIncrement
from Methods.batchmodel import Kmeans, Kmedoids, KShapes, DTCluster
from Evaluation import purity
from Methods.streaming import DenStream, MiniBatchKMeans, STREAMLS, Clustream2, DbStream
from Methods.streaming import BirchN, BirchK
from utils import get_dataset, get_dataset_noisy


def run_Clustering_on_noisy(method_name, online, dataset_name, modified_params={}, init_size=None):
    global Parameters
    global Need_K
    for param in modified_params.keys():
        Parameters[method_name][param] = modified_params[param]
    if online:
        function_name = f'Online_{method_name}'
    else:
        function_name = f'Static_{method_name}'
    function_to_call = globals()[function_name]

    fitX, fity, PredX, Predy, _ = get_dataset_noisy(dataset_name, init_size=init_size)
    data = [fitX, fity, PredX, Predy]
    k_in_initial_batch = len(set(fity))
    if method_name in Need_K:
        results = function_to_call(data, k_in_initial_batch, **Parameters[method_name])
    else:
        results = function_to_call(data, **Parameters[method_name])
    return results


def run_Clustering(method_name, online, dataset_name, modified_params={}, init_size=None):
    global Parameters
    global Need_K
    for param in modified_params.keys():
        Parameters[method_name][param] = modified_params[param]
    if online:
        function_name = f'Online_{method_name}'
    else:
        function_name = f'Static_{method_name}'
    function_to_call = globals()[function_name]

    fitX, fity, PredX, Predy, _ = get_dataset(dataset_name, init_size=init_size)
    data = [fitX, fity, PredX, Predy]
    k_in_initial_batch = len(set(fity))
    if method_name in Need_K:
        results = function_to_call(data, k_in_initial_batch, **Parameters[method_name])
    else:
        results = function_to_call(data, **Parameters[method_name])
    return results

    # except Exception as  e:
    #     print(e)
    #     print(f'Error during running {method_name} on {dataset_name} dataset')
    #     return 'Error'


def evaluation_metrics(pred, real):
    res_purity = purity(pred, real)
    ri = rand_score(pred, real)
    ari = adjusted_rand_score(pred, real)
    nmi = normalized_mutual_info_score(pred, real)
    return {"Purity": res_purity, "RI": ri, "ARI": ari, "NMI": nmi}


def apply_online_clustering(data, cluster_method):
    fitX, fity, PredX, Predy = data[0], data[1], data[2], data[3]
    clf = cluster_method
    start = time.time()
    clf.fit(fitX, fity)
    predictions = clf.predict(PredX)
    Runtime = time.time() - start
    pred = [pr + 1 for pr in predictions]
    real = [pr for pr in Predy]
    eval_metrics = evaluation_metrics(pred, real)
    eval_metrics["Runtime"] = Runtime
    return eval_metrics


def apply_static_clustering(data, cluster_method):
    fitX, fity, PredX, Predy = data[0], data[1], data[2], data[3]
    clf = cluster_method
    start = time.time()
    clf.fit(np.vstack([fitX, PredX]), fity + Predy)
    predictions = clf.predict(PredX)
    Runtime = time.time() - start
    pred = [pr + 1 for pr in predictions]
    real = [pr for pr in Predy]
    eval_metrics = evaluation_metrics(pred, real)
    eval_metrics["Runtime"] = Runtime
    return eval_metrics


def Static_KShapes(data, k, distance_measure="euclidean", transformation="default"):
    clf = KShapes(k=k, distance_measure="SBD", transformation="default")
    eval_metrics = apply_static_clustering(data, clf)
    return eval_metrics


def Static_Kmeans(data, k, distance_measure="euclidean", transformation="default"):
    clf = Kmeans(k=k, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_static_clustering(data, clf)
    return eval_metrics


def Static_Kmedoids(data, k, distance_measure="SBD", transformation="default"):
    clf = Kmedoids(k=k, distance_measure=distance_measure, transformation=transformation)
    eval_metrics = apply_static_clustering(data, clf)
    return eval_metrics


def Static_DTC(data, k, distance_measure="euclidean", transformation="default"):
    clf = DTCluster(k=k, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_static_clustering(data, clf)
    return eval_metrics


def Online_Kmeans(data, k, distance_measure="euclidean", transformation="default"):
    clf = Kmeans(k=k, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_Kmedoids(data, k, distance_measure="SBD", transformation="default"):
    clf = Kmedoids(k=k, distance_measure=distance_measure, transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_TKmeans(data, k, distance_measure="euclidean", transformation="default"):
    clf = trivialIncrement.Kmeans(k=k, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_TKmedoids(data, k, distance_measure="SBD", transformation="default"):
    clf = trivialIncrement.Kmedoids(k=k, distance_measure=distance_measure, transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_TKShapes(data, k, distance_measure="euclidean", transformation="default"):
    clf = trivialIncrement.KShapes(k=k, distance_measure="SBD", transformation="default")
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_DTC(data, k, distance_measure="euclidean", transformation="default"):
    clf = DTCluster(k=k, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_KShapes(data, k, distance_measure="euclidean", transformation="default"):
    clf = KShapes(k=k, distance_measure="SBD", transformation="default")
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_DenStream(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = DenStream(slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_MiniBatchKMeans(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = MiniBatchKMeans(k, slide, distance_measure="euclidean", transformation=transformation)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_STREAMLS(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = STREAMLS(k, slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_Clustream(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = Clustream2(k, slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_DbStream(data, distance_measure="euclidean", transformation="catch22", **params):
    slide = len(data[0])
    clf = DbStream(slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BirchN(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BirchN(slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BirchK(data, k, distance_measure="euclidean", transformation="catch22", **params):
    slide = len(data[0])
    clf = BirchK(k=k, slide=slide, distance_measure="euclidean", transformation=transformation, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURST_Clustering(slide, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_L(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURST_Clustering_L(slide, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_NaiveMerge(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURST_Clustering_NaiveMerge(slide, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_2S(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURST_Clustering_2S(slide, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURSTK(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURSTK_Clustering(slide, k, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_SNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURSTK_Clustering(slide, 2, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_LNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURSTK_Clustering(slide, int(sqrt(slide)), **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_LNK_NaiveMerge(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURSTK_Clustering_NaiveMerge(slide, int(sqrt(slide)), **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_SNK_NaiveMerge(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    clf = BURSTK_Clustering_NaiveMerge(slide, 2, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_GEN_KMeans(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN(baseline, slide, k=None, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_GEN_KMeans_NaiveMerge(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN_NaiveMerge(baseline, slide, k=None, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_GEN_KMeans_2S(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN_2S(baseline, slide, k=None, distance_measure=distance_measure, transformation=transformation,
                       verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_GEN_KMedoids(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmedoids
    clf = BURST_GEN(baseline, slide, k=None, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_GEN_KMedoids_2S(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmedoids
    clf = BURST_GEN_2S(baseline, slide, k=None, distance_measure=distance_measure, transformation=transformation,
                       verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics



def Online_BURST_GEN_KMeans_SNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN(baseline, slide, k=2, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


def Online_BURST_GEN_KMeans_LNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN(baseline, slide, k=int(sqrt(slide)), distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics



def Online_BURSTK_GEN_KMeans(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmeans
    clf = BURST_GEN(baseline, slide, k=k, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_GEN_KMedoids_SNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmedoids
    clf = BURST_GEN(baseline, slide, k=2, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURST_GEN_KMedoids_LNK(data, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmedoids
    clf = BURST_GEN(baseline, slide, k=int(sqrt(slide)), distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics

def Online_BURSTK_GEN_KMedoids(data, k, distance_measure="euclidean", transformation="default", **params):
    slide = len(data[0])
    baseline = Kmedoids
    clf = BURST_GEN(baseline, slide, k=k, distance_measure=distance_measure, transformation=transformation,
                    verbose=False, **params)
    eval_metrics = apply_online_clustering(data, clf)
    return eval_metrics


Need_K = ["KShapes", "Kmeans", "Kmedoids", "DTC", "MiniBatchKMeans", "STREAMLS", "Clustream", "BirchK", "BURSTK",
          "BURSTK_GEN_KMeans", "BURSTK_GEN_KMedoids", "TKShapes", "TKmedoids", "TKmeans"]

Parameters = {
    # parameters used except k, slide, distance metric and transformation.
    "KShapes": {"distance_measure": "SBD", "transformation": "default"},
    "Kmeans": {"distance_measure": "euclidean", "transformation": "default"},
    "Kmedoids": {"distance_measure": "SBD", "transformation": "default"},
    "DenStream": {'decaying_factor': 0.01, 'beta': 1, 'mu_factor': 5, 'epsilon': 1, "distance_measure": "euclidean",
                  "transformation": "default"},
    "MiniBatchKMeans": {"distance_measure": "euclidean", "transformation": "default"},
    "STREAMLS": {"halflife": 1, "distance_measure": "euclidean", "transformation": "catch22"},
    "Clustream": {"radiusfactor": 1, "k_multiplier": 2, "distance_measure": "euclidean", "transformation": "catch22"},
    "DbStream": {"fading_factor": 0.05, "cleanup_interval": 0.8, "intersection_factor": 0.2,
                 "clustering_threshold": 1.5, "minimum_weight": 1, "distance_measure": "euclidean",
                 "transformation": "catch22"},
    "BirchN": {"threshold": 0.5, "distance_measure": "euclidean", "transformation": "default"},
    "BirchK": {"threshold": 0.5, "distance_measure": "euclidean", "transformation": "catch22"},
    "BURST": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURSTK": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURST_GEN_KMeans": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURST_GEN_KMedoids": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURSTK_GEN_KMeans": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURSTK_GEN_KMedoids": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "DTC": {"distance_measure": "euclidean", "transformation": "default"},  # Default from Odyssey

    "TKShapes": {"distance_measure": "SBD", "transformation": "default"},
    "TKmeans": {"distance_measure": "euclidean", "transformation": "default"},
    "TKmedoids": {"distance_measure": "SBD", "transformation": "default"},

    "BURST_GEN_KMeans_2S": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURST_2S": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURST_GEN_KMedoids_2S": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},

    "BURST_GEN_KMeans_SNK": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURST_SNK": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURST_GEN_KMedoids_SNK": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},

    "BURST_GEN_KMeans_LNK": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURST_LNK": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURST_GEN_KMedoids_LNK": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},

    "BURST_GEN_KMeans_NaiveMerge": {"alpha": 0.3, "distance_measure": "euclidean", "transformation": "default"},
    "BURST_NaiveMerge": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},

    "BURST_LNK_NaiveMerge": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
    "BURST_SNK_NaiveMerge": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},

    "BURST_L": {"alpha": 0.3, "distance_measure": "SBD", "transformation": "default"},
}
