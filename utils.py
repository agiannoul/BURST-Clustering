import os
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def load_generate_synthetic_data(init_per=None):
    patter_one=[i for i in range(1,10)]+[10-i for i in range(1,10)]+[i for i in range(1,10)]+[10-i for i in range(1,10)]

    pattern_two=[i for i in range(1,10)]+[10-i for i in range(1,10)]+[-i for i in range(1,10)]+[-10+i for i in range(1,10)]

    pattern_three=[i for i in range(1,10)]+[10-i for i in range(1,5)]+[5 for i in range(1,20)]+[5-i for i in range(1,5)]
    pattern_four=[10-i for i in range(1,10)]+[i for i in range(1,10)]+[10-i for i in range(1,10)]+[i for i in range(1,10)]

    # plt.subplot(311)
    # plt.plot(patter_one)
    # plt.subplot(312)
    # plt.plot(pattern_two)
    # plt.subplot(313)
    # plt.plot(pattern_three)
    # plt.show()
    data=[]
    labels=[]
    a=0.1
    b=1
    for i in range(100):
        data.append([pt+ random.uniform(a, b) for pt in patter_one])
        labels.append(1)
        data.append([pt+ random.uniform(a, b) for pt in pattern_two])
        labels.append(2)
        data.append([pt+ random.uniform(a, b) for pt in pattern_three])
        labels.append(3)
    for i in range(100):
        data.append([pt+ random.uniform(a, b) for pt in patter_one])
        labels.append(1)
        data.append([pt+ random.uniform(a, b) for pt in pattern_two])
        labels.append(2)
        data.append([pt+ random.uniform(a, b) for pt in pattern_three])
        labels.append(3)
        data.append([pt + random.uniform(a, b) for pt in pattern_four])
        labels.append(4)
    data=np.array(data).astype(float)
    if init_per is None:
        init_size = int(0.1 * len(labels))
    else:
        init_size = max(20,int(init_per * len(labels)))
    return data[:init_size], labels[:init_size], data[init_size:], labels[init_size:], init_size


def load_generate_2_3_2new(init_per=None,length_factor=10,size_factor=100):
    length_factor=length_factor
    size_factor=size_factor
    total_size=size_factor*7
    total_length=(length_factor-1)*4
    patter_one=[i for i in range(1,length_factor)]+[length_factor-i for i in range(1,length_factor)]+[i for i in range(1,length_factor)]+[length_factor-i for i in range(1,length_factor)]

    pattern_two=[i for i in range(1,length_factor)]+[length_factor-i for i in range(1,length_factor)]+[-i for i in range(1,length_factor)]+[-length_factor+i for i in range(1,length_factor)]

    pattern_three=[i for i in range(1,length_factor)]+[length_factor-i for i in range(1,length_factor//2)]+[5 for i in range(1,length_factor*2)]+[5-i for i in range(1,length_factor//2)]
    pattern_four=[10-i for i in range(1,length_factor)]+[i for i in range(1,length_factor)]+[length_factor-i for i in range(1,length_factor)]+[i for i in range(1,length_factor)]

    np.random.seed(0)
    time = np.linspace(0, 10,total_length)
    freq1 = 1  # Frequency of 1 Hz
    freq2 = 3  # Frequency of 3 Hz

    phase1_signal2 = np.pi / 4  # Different phases for Signal 2
    phase2_signal2 = -np.pi / 4
    pattern_five = np.sin(2 * np.pi * freq1 * time + phase1_signal2) + np.sin(2 * np.pi * freq2 * time + phase2_signal2)

    # plt.subplot(311)
    # plt.plot(patter_one)
    # plt.subplot(312)
    # plt.plot(pattern_two)
    # plt.subplot(313)
    # plt.plot(pattern_three)
    # plt.show()
    data=[]
    labels=[]
    a=0.1
    b=1
    for i in range(size_factor):
        data.append([pt+ random.uniform(a, b) for pt in patter_one])
        labels.append(1)
        data.append([pt+ random.uniform(a, b) for pt in pattern_two])
        labels.append(2)
    for i in range(size_factor):
        data.append([pt+ random.uniform(a, b) for pt in patter_one])
        labels.append(1)
        data.append([pt+ random.uniform(a, b) for pt in pattern_two])
        labels.append(2)
        data.append([pt+ random.uniform(a, b) for pt in pattern_three])
        labels.append(3)
    for i in range(size_factor):
        data.append([pt+ random.uniform(a, b) for pt in pattern_four])
        labels.append(4)
        data.append([pt + random.uniform(a, b) for pt in pattern_five])
        labels.append(5)
    data=np.array(data).astype(float)
    if init_per is None:
        init_size = int(0.1 * len(labels))
    else:
        init_size = max(20,int(init_per * len(labels)))
    return data[:init_size], labels[:init_size], data[init_size:], labels[init_size:], init_size


def apply_clustering(fitX,fity,PredX,Predy,cluster_method):
    clf=cluster_method
    clf.fit(fitX,fity)
    predicitons=clf.predict(PredX)
    return [pr+1 for pr in predicitons],[pr for pr in Predy]


def apply_clustering_stream(fitX,fity,PredX,Predy,cluster_method):
    clf=cluster_method
    clf.fit(fitX)
    #print(clf.model.n_clusters)
    predicitons=clf.predict(PredX)
    return [pr+1 for pr in predicitons],[pr for pr in Predy]


def get_ucr_dataset(datasetname,init_per=None):
    datasetname=datasetname.split("_")[-1]
    dftr = pd.read_csv(f"UCR_DATASETS/{datasetname}/{datasetname}_TRAIN", header=None)
    dfte = pd.read_csv(f"UCR_DATASETS/{datasetname}/{datasetname}_TEST", header=None)
    df = pd.concat([dftr, dfte], axis=0, ignore_index=True)
    labels = df[df.columns[0]].values
    data = df.drop([df.columns[0]], axis=1).values

    # print(labels)
    # print(data)
    if init_per is None:
        init_size = int(0.1 * len(labels))
    else:
        init_size = int(init_per * len(labels))
    return data[:init_size], labels[:init_size], data[init_size:], labels[init_size:], init_size

def get_dataset(datasetname,init_size=None):
    if datasetname=="generate_synthetic_data":
        return load_generate_synthetic_data(init_size)
    elif "generate_2_3_2new" in datasetname:
        lenf=10
        sizf=100
        #generate_2_3_2new_lenf_10_sizf_100
        if "lenf" in datasetname:
            lenf=int(datasetname.split("lenf")[1].split("_")[1])
        if "sizf" in datasetname:
            sizf=int(datasetname.split("sizf")[1].split("_")[1])
        return load_generate_2_3_2new(init_size,length_factor=lenf,size_factor=sizf)
    elif "UCR_" in datasetname:
        return get_ucr_dataset(datasetname,init_size)



