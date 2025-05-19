import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import read_mehtod



def module_impact():
    tests=["2S","SNK","NaiveMerge","SNK_NaiveMerge","L"]
    message=["Replace 2T with 2S","k=2","No merging","No merging and k=2","replace complete distance with single linkage"]
    method_to_call=[show_performance_diff_2S,show_performance_diff_SNK,show_performance_diff_SNK,show_performance_diff_SNK,show_performance_diff_SNK]
    # tests = []
    # method_to_call = [show_performance_diff_SNK]
    method_name="BURST"
    res={}
    for test,get_res in zip(tests,method_to_call):
        originalm=method_name
        new=method_name+"_"+test
        metric="Purity"
        wins2,p_value2,original2,newm2,less2=get_res(originalm,new,metric)
        percentage2 = [(nm-ori)  for ori, nm in zip(original2, newm2) if ori != 0]
        orig2 = [(ori) for ori, nm in zip(original2, newm2) if ori != 0]
        metric="ARI"
        wins,p_value,original,newm,less=get_res(originalm,new,metric)
        # print(p_value)
        # print(wins)
        percentage=[(nm-ori)  for ori, nm in zip(original, newm) if ori != 0]
        orig1=[(ori)  for ori, nm in zip(original, newm) if ori != 0]
        if p_value2<0.05:
            impact2="+"
        elif less2<0.05:
            impact2="-"
        else:
            impact2="="
        if p_value < 0.05:
            impact = "+"
        elif less < 0.05:
            impact = "-"
        else:
            impact = "="
        res[test] = [impact,np.mean(percentage)/np.mean(orig1),impact2,np.mean(percentage2)/np.mean(orig2)]
    print("test, ARI, Purity")
    for key in res.keys():
        print(f"{message[tests.index(key)]}: {res[key][0]}, {res[key][1]}|{res[key][2]}, {res[key][3]}")






def show_performance_diff_2S(original,new,metric):
    dfor = read_mehtod("resultsUP.csv",[original])


    df= pd.read_csv("ablation_S.csv")
    df= df[df["method_name"]==new]
    dfor=dfor[dfor["dataset_name"].isin(df["dataset_name"])]


    # print(dfor.shape)
    # print(df.shape)


    df.sort_values(by=["dataset_name"], inplace=True)
    dfor.sort_values(by=["dataset_name"], inplace=True)

    greater = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm>dfm])
    equal = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm==dfm])
    less = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm<dfm])

    # print(f"Greater: {greater}, Equal: {equal}, Less: {less}")
    from scipy.stats import wilcoxon
    res = wilcoxon(dfor[metric], df[metric], alternative="greater")
    p_value = res.pvalue
    res = wilcoxon(dfor[metric], df[metric], alternative="less")
    p_valueless = res.pvalue
    # print(p_value)
    # if p_value < 0.05:
    #     print("dfor is statistically better than df (p < 0.05).")
    # else:
    #     print("No significant difference (p >= 0.05).")

    # print(dfor[metric].mean())
    # print(df[metric].mean())
    return [greater, equal, less], p_value, dfor[metric], df[metric],p_valueless
def show_performance_diff_SNK(original,new,metric):
    dfor = read_mehtod("resultsUP.csv",[original])


    df= pd.read_csv("ablation_SNK.csv")
    df= df[df["method_name"]==new]
    df=df.drop_duplicates(subset=["dataset_name"], keep="last")
    dfor=dfor[dfor["dataset_name"].isin(df["dataset_name"])]


    # print(dfor.shape)
    # print(df.shape)


    df.sort_values(by=["dataset_name"], inplace=True)
    dfor.sort_values(by=["dataset_name"], inplace=True)

    greater = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm>dfm])
    equal = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm==dfm])
    less = sum([1  for dorm,dfm in zip(dfor[metric].values,df[metric].values) if dorm<dfm])

    # print(f"Greater: {greater}, Equal: {equal}, Less: {less}")
    from scipy.stats import wilcoxon
    res = wilcoxon(dfor[metric], df[metric], alternative="greater")
    p_value=res.pvalue
    res = wilcoxon(dfor[metric], df[metric], alternative="less")
    p_valueless = res.pvalue
    # print(p_value)
    # if p_value < 0.05:
    #     print("dfor is statistically better than df (p < 0.05).")
    # else:
    #     print("No significant difference (p >= 0.05).")

    # print(dfor[metric].mean())
    # print(df[metric].mean())

    return [greater,equal,less],p_value,dfor[metric],df[metric],p_valueless
    # plt.scatter(dfor[metric],df[metric],marker="s")
    # plt.plot([0,1],[0,1])
    # plt.show()

module_impact()


