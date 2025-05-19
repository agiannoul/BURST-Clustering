import os
from random import random
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import time
from Evaluation import purity
from sklearn.metrics import rand_score, normalized_mutual_info_score, adjusted_rand_score

def load_generate_synthetic_data(init_per=None):
    patter_one=[i for i in range(1,10)]+[10-i for i in range(1,10)]+[i for i in range(1,10)]+[10-i for i in range(1,10)]

    pattern_two=[i for i in range(1,10)]+[10-i for i in range(1,10)]+[-i for i in range(1,10)]+[-10+i for i in range(1,10)]

    pattern_three=[i for i in range(1,10)]+[10-i for i in range(1,5)]+[5 for i in range(1,20)]+[5-i for i in range(1,5)]
    pattern_four=[10-i for i in range(1,10)]+[i for i in range(1,10)]+[10-i for i in range(1,10)]+[i for i in range(1,10)]

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

def load_data1(init_per=None):
    dftr=pd.read_csv("UCR_DATASETS/Trace_TRAIN",header=None)
    dfte=pd.read_csv("UCR_DATASETS/Trace_TEST",header=None)
    df=pd.concat([dftr,dfte],axis=0,ignore_index=True)
    labels=df[df.columns[0]].values
    data=df.drop([df.columns[0]], axis=1).values

    # print(labels)
    # print(data)
    if init_per is None:
        init_size = int(0.1 * len(labels))
    else:
        init_size=int(init_per*len(labels))
    return data[:init_size],labels[:init_size],data[init_size:],labels[init_size:],init_size

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

def get_ucr_dataset_noisy(datasetname,init_per=None):
    datasetname=datasetname.split("_")[-1]
    dftr = pd.read_csv(f"Noisy_UCR_DATASETS/{datasetname}/{datasetname}_TRAIN", header=None)
    dfte = pd.read_csv(f"Noisy_UCR_DATASETS/{datasetname}/{datasetname}_TEST", header=None)
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


def get_dataset_noisy(datasetname,init_size=None):
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
        return get_ucr_dataset_noisy(datasetname,init_size)





def apply_method_k_init(dataset_name,model,verbose=False,write_res=True):
    fitX, fity, PredX, Predy, init_size = get_dataset(dataset_name)
    k = len(list(set(fity)))
    # model = StreamK(Kmeans, len(fity), alpha=0.3, distance_measure='euclidean',
    #                 verbose=verbose)
    ari=recc_method_k(fitX, fity, PredX, Predy, init_size, dataset_name, "SCKmeans", model,
                  verbose=verbose, write_res=write_res)


def recc_method_k(fitX, fity, PredX, Predy, init_siz,dataset_name,method_name,
                  methodclass,verbose=False,write_res=True,filename_to_save='results.csv'):

    start=time.time()
    pred, real = apply_clustering(fitX, fity, PredX, Predy,methodclass)
    endtime=time.time()-start

    number_of_classes_in_fit=len(list(set(fity)))
    alllabels=[lb for lb in fity]
    alllabels.extend(Predy)
    number_of_classes_in_total=len(list(set(alllabels)))

    res_purity=purity(pred,real)
    ri = rand_score(pred, real)
    ari = adjusted_rand_score(pred, real)
    nmi = normalized_mutual_info_score(pred, real)
    exeTime = endtime

    parameter_names=[]
    parameter_values=[]
    parameters=methodclass.get_parameters()
    for key in parameters:
        parameter_names.append(key)
        parameter_values.append(parameters[key])

    parameters_name_field="["+"|".join(parameter_names)+"]"
    parameters_value_field="["+"|".join([str(v) for v in parameter_values])+"]"

    # columns: datasetname,method_name,purity,ari,ri,nmi,Time,init_size,number_of_classes_in_fit,number_of_classes_in_total,parameters_name_field,parameters_value_field


    if verbose:
        print(f" {method_name}")
        print(" Purity Score:",res_purity)
        print(' Adjusted Rand Score:', ari)
        print(' Rand Score:', ri)
        print(' Normalized Mutual Information:', nmi)
        print(' Time', exeTime)
        print(" parameters: ",parameters)
        print(' Classes in fit', number_of_classes_in_fit)
        print(' Classes (all)', number_of_classes_in_total)
        print('==========================================')

    row=dataset_name
    #["method_name","Purity","ARI","RI","NMI","Runtime","Fit_size","number_of_classes_in_fit","number_of_classes_in_total","parameters_names","parameters_value"]
    for colvalue in [method_name,res_purity,ari,ri,nmi,exeTime,init_siz,number_of_classes_in_fit,number_of_classes_in_total,parameters_name_field,parameters_value_field]:
        row+=","+str(colvalue)
    if write_res:
        with open(filename_to_save, 'a') as file:
            file.write(row+"\n")





def expRun(methodname,filename,exclude=[],modified_params={}):
    import allMethods
    online=True
    init_size=None
    dfd = pd.read_csv('resultsUP.csv')
    datasets=dfd["dataset_name"].unique()
    # init_size=1000
    # init_size=3500
    repeats=1
    exec_time=[]
    lengths=[]
    counti=0
    for name in tqdm(datasets):
        try:
            if name in exclude:
                continue
            runtimeall=0
            for i in range(repeats):
                counti+=1
                res = allMethods.run_Clustering(methodname, online, name,init_size=init_size,modified_params=modified_params)
                runtimeall+=res["Runtime"]
                ARI=res["ARI"]
                Purity=res["Purity"]
                RI=res["RI"]
                NMI=res["NMI"]
            print(runtimeall/repeats)
            exec_time=runtimeall/repeats
            append_to_trivial_incr([name,methodname,ARI,Purity,RI,NMI,exec_time],filename)
        except Exception as e:
            print(e)
            continue


def expRun_noisy(methodname,filename,exclude=[]):
    import allMethods
    online=True
    init_size=None
    dfd = pd.read_csv('resultsUP.csv')
    datasets=dfd["dataset_name"].unique()
    # init_size=1000
    # init_size=3500
    repeats=1
    exec_time=[]
    lengths=[]
    counti=0
    for name in tqdm(datasets):
        try:
            if name in exclude:
                continue
            runtimeall=0
            for i in range(repeats):
                counti+=1
                res = allMethods.run_Clustering_on_noisy(methodname, online, name,init_size=init_size)
                runtimeall+=res["Runtime"]
                ARI=res["ARI"]
                Purity=res["Purity"]
                RI=res["RI"]
                NMI=res["NMI"]
            print(runtimeall/repeats)
            exec_time=runtimeall/repeats
            append_to_trivial_incr([name,methodname,ARI,Purity,RI,NMI,exec_time],filename)
        except Exception as e:
            print(e)
            continue

def append_to_trivial_incr(variables,filename):
    """
    Appends a list of variables as a single row to the file 'trivial_incr.csv'.

    Parameters:
        variables (list): List of variables to append.
    """
    with open(filename, "a") as file:  # 'a' mode ensures the file is created if it doesn't exist
        line = ",".join(map(str, variables)) + "\n"
        file.write(line)

def read_mehtod(filename,keepmethods,parameters=None):
    df = pd.read_csv(filename)
    df = df[df["method_name"].isin(keepmethods)]
    if parameters is not None:
        df = df[df["parameters_value"].str.contains(parameters)]
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                "parameters_names", "parameters_value"], keep='last')
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                ], keep='first')
    return df


