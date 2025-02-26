import statistics
import autorank
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

import seaborn as sns
def violinplot2(build_df,metric,ax,colordict_f=None):

    allnames=[]
    allscores=[]

    todfdic={"name":[],f'{metric}':[]}
    for key in build_df.keys():
        todfdic["name"].extend([key for v in build_df[key]])
        todfdic[f'{metric}'].extend([v for v in build_df[key]])
        allnames.append(key)
        allscores.append(build_df[key])
    # Sort data, labels, and colors by median values

    medians = [np.median(lst) for lst in allscores]
    sorted_indices = np.argsort(medians)
    allnames = [allnames[i] for i in sorted_indices]

    if colordict_f is None:
        snscolors = sns.color_palette("coolwarm", len(sorted_indices))
    else:
        snscolors = [colordict_f[name] for name in allnames]

    custom_palette = {method: color for method, color in zip(allnames, snscolors)}

    df=pd.DataFrame(todfdic)
    df['name'] = pd.Categorical(df['name'], categories=allnames, ordered=True)
    df = df.sort_values('name')
    sns.boxplot(data=df,x="name", y=metric, showfliers=False,
                meanprops=dict(color='k', linestyle='--'), showmeans=True, meanline=True, palette=custom_palette,ax=ax)

    ax.set_xticks(range(len(allnames)), labels=allnames,rotation=30,ha='right')
    sep_position = len(allnames) - 1  # Position before the last category
    ax.set_ylabel(metric,fontsize=14)
    ax.set_xlabel("")


def violinplot(build_df,metric,ax,colordict_f=None):

    allnames=[]
    allscores=[]

    todfdic={"name":[],f'{metric}':[]}
    for key in build_df.keys():
        todfdic["name"].extend([key for v in build_df[key]])
        todfdic[f'{metric}'].extend([v for v in build_df[key]])
        allnames.append(key)
        allscores.append(build_df[key])
    # Sort data, labels, and colors by median values

    medians = [np.median(lst) for lst in allscores]
    sorted_indices = np.argsort(medians)
    allnames_temp = [allnames[i] for i in sorted_indices]
    allnames=[""]
    for i in range(len(allnames_temp)):
        if "Static" in allnames_temp[i]:
            allnames = [allnames_temp[i] for q in range(len(allnames_temp))]
            break
    counter=0
    for i in range(len(allnames)):
        if "Static" in allnames_temp[i]:
            continue
        allnames[counter] = allnames_temp[i]
        counter+=1
    if colordict_f is None:
        snscolors = [sns.color_palette("coolwarm", len(sorted_indices))]
    else:
        snscolors = [colordict_f[name] for name in allnames]

    custom_palette = {method: color for method, color in zip(allnames, snscolors)}

    df=pd.DataFrame(todfdic)
    df['name'] = pd.Categorical(df['name'], categories=allnames, ordered=True)
    df = df.sort_values('name')
    sns.boxplot(data=df,x="name", y=metric, showfliers=False,
                meanprops=dict(color='k', linestyle='--'), showmeans=True, meanline=True, palette=custom_palette,ax=ax)

    ax.set_xticks(range(len(allnames)), labels=allnames, fontsize=10,rotation=30,ha='right')
    sep_position = len(allnames) - 1  # Position before the last category
    ax.axvline(x=sep_position - 0.5, color='black', linestyle='--', linewidth=2)
    ax.set_ylabel(metric,fontsize=14)
    ax.set_xlabel("")

def rankmodelspersetting(dfdictini,titletext,ax):

    #matplotlib.rc('font', **font)
    dfdict={}
    for key in dfdictini.keys():
        dfdict[key.replace("\n","_")]=dfdictini[key]
    df2 = pd.DataFrame(dfdict)
    print(df2.shape[0])
    results = autorank.autorank(df2,
                                alpha=0.05,
                                verbose=False,
                                order='descending',
                                #force_mode='nonparametric'
                                )
    autorank.create_report(results)
    ax=autorank.plot_stats(results,width=4,ax=ax)

    return ax

def plot_percentages(percentages,methods,symbols,ax,title,colors = ['#34a0e2', '#eccda4', '#eae6df']):
    import matplotlib.pyplot as plt
    import numpy as np

    # Data from the list
    # methods = [
    #     'CNN', 'KMeansAD', 'KShapeAD', 'LSTMAD', 'MOMENT', 'MatrixProfile', 'POLY', 'Sub_PCA', 'TimesFM', 'USAD'
    # ]
    # percentages  = [
    #     (49.41, 19.89, 30.70),  # CNN
    #     (27.23, 47.36, 25.41),  # KMeansAD_U
    #     (19.75, 42.88, 37.37),  # KShapeAD
    #     (42.95, 26.51, 30.54),  # LSTMAD
    #     (41.14, 30.46, 28.40),  # MOMENT
    #     (34.72, 33.34, 31.94),  # MatrixProfile
    #     (33.88, 40.69, 25.43),  # POLY
    #     (32.37, 42.45, 25.18),  # Sub_PCA
    #     (58.19, 14.65, 27.15),  # TimesFM
    #     (29.22, 48.73, 22.05),  # USAD
    # ]
    # symbols = [
    #     '✔', '✔', '✘', '✔', '✔', '✔', '✔', '✔', '✔', '✔'  # Check and Cross symbols
    # ]
    # Colors for each segment: green, grey, orange
    for i in range(len(percentages)):
        if percentages[i][1]<1:
            percentages[i]=(percentages[i][0]-1,2, percentages[i][2]-1)
    sorted_data = sorted(zip(methods, percentages, symbols), key=lambda x: x[0])#, reverse=False)

    methods, percentages, symbols = zip(*sorted_data)

    # Prepare the figure and axis

    # Plot each method as a horizontal bar
    y_pos = np.arange(len(methods))

    # Plot each method as a horizontal bar
    for i, (green, grey, orange) in enumerate(percentages):
        # Plot the segments
        ax.barh(i, green, color=colors[0], label="Wins" if i == 0 else "")
        ax.barh(i, grey, left=green, color=colors[1], label="Equal" if i == 0 else "")
        ax.barh(i, orange, left=green + grey, color=colors[2], label="Worst" if i == 0 else "")

        # Add symbols to the end of the bar
        if symbols[i]=='  ✔' or symbols[i]=='  ✘':
            ax.text(green + grey + orange + 1, i, symbols[i], va='center', ha='left', fontsize=14)

        # Add percentage labels in the middle of each segment (without decimals)
        ax.text(green / 2, i, f'{int(green)}%', va='center', ha='center', color='black', fontsize=12)
        ax.text(green + grey *0.1, i, f'{int(grey)}%', va='center', ha='center', color='black', fontsize=12)
        ax.text(green + grey + orange*0.8, i, f'{int(orange)}%', va='center', ha='center', color='black', fontsize=12)

    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_xticks([])
    ax.set_yticklabels(methods,fontsize=14)
    # ax.set_xlabel("Performance (%)")
    ax.set_title(title,fontsize=14)

    # Add legend
    import matplotlib.patches as mpatches

    green_patch = mpatches.Patch(color=colors[0], label='Wins')
    blue_patch = mpatches.Patch(color=colors[1], label='Equal')
    orange_patch = mpatches.Patch(color=colors[2], label='Worse')

    handles = [
        green_patch,
        blue_patch,
        orange_patch,
        plt.Line2D([0], [0], color="white", label="✔ statistically significant better"),
    ]
    # ax.legend(
    #     handles=handles,
    #     loc='upper center',
    #     ncols=5,
    #     bbox_to_anchor=(0.5, 1.2)
    # )
    # Show the plot
    # plt.tight_layout()
    # plt.show()



def find_greater_k(rows):
    row_with_m_k=None
    maxk=-1
    for ind,row in rows.iterrows():
        kpos=row["parameters_names"]
        pameters=kpos.replace("[","").replace("]","").split("|")
        if "k" not in pameters:
            return row
        kpos=pameters.index("k")
        tempk=int(row["parameters_value"].replace("[","").replace("]","").split("|")[kpos])
        if tempk>maxk:
            row_with_m_k=row
            maxk=tempk
    return row_with_m_k
def find_min_k(rows):
    row_with_m_k=None
    maxk=10000000000000000000000000
    for ind, row in rows.iterrows():
        kpos = row["parameters_names"]
        pameters = kpos.replace("[", "").replace("]", "").split("|")
        if "k" not in pameters:
            return row
        kpos = pameters.index("k")
        tempk = int(row["parameters_value"].replace("[","").replace("]","").split("|")[kpos])
        if tempk<maxk:
            row_with_m_k=row
            maxk=tempk
    return row_with_m_k


def find_instance(dfdata,vers,type):
    dfm1 = dfdata[dfdata['method_name'] == vers.split("_")[0]]
    if "_" in vers:
        if vers.split("_")[1] == "22":
            dfm1 = dfm1[dfm1["parameters_value"].str.contains("catch22")]
        elif vers.split("_")[1] == "SBD":
            dfm1 = dfm1[dfm1["parameters_value"].str.contains("SBD")]
    else:
        dfm1 = dfm1[~dfm1["parameters_value"].str.contains("catch22")]
        dfm1 = dfm1[~dfm1["parameters_value"].str.contains("SBD")]

    if type=="init":
        frow=find_min_k(dfm1)
    else:
        frow=find_greater_k(dfm1)
    return frow

def Burst_Impact(df,ax,title,metric="ARI",dtype="evol",colors = ['#34a0e2', '#eccda4', '#eae6df']):
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                "parameters_names", "parameters_value"], keep='last')
    if dtype=="evol":
        df = df[df['number_of_classes_in_fit'] != df['number_of_classes_in_total']]
    elif dtype=="notevol":
        df = df[df['number_of_classes_in_fit'] == df['number_of_classes_in_total']]


    versus=[("Kmedoids_22","SCKmedoidsK_22"),("Kmedoids","SCKmedoidsK"),("Kmedoids_SBD","SCKmedoidsK_SBD")
        ,("Kmeans_22", "SCKmeansK_22"), ("Kmeans", "SCKmeansK")]
    #versus = [("Kmeans_22", "SCKmeansK_22"), ("Kmeans", "SCKmeansK"),
    #          ("Kmeans_22", "SCKmeans_22"), ("Kmeans", "SCKmeans")]
    #versus = [("SCKmeansK_22", "SCKmeans_22"), ("SCKmeansK", "SCKmeans")]

    methods=[]
    percentages=[]
    symbols=[]
    for vers in versus:
        m1valus=[]
        m2valus=[]
        wins=0
        loses=0
        equal=0
        for dataset in df["dataset_name"].unique():
            for ktype in ["init"]:
                dfdata=df[df["dataset_name"]==dataset]
                # print(dfdata.head())
                m1=find_instance(dfdata,vers[0],ktype)[metric]
                m2=find_instance(dfdata,vers[1],ktype)[metric]
                m1valus.append(m1)
                m2valus.append(m2)
                if m1>m2:
                    loses+=1
                elif m2>m1:
                    wins+=1
                else:
                    equal+=1
        res = wilcoxon(m2valus,m1valus, alternative='greater')
        if res[1] <= 0.05:
            better = "\\greencheck"
            symbols.append('  ✔')
        else:
            res = wilcoxon(m2valus, m1valus, alternative='less')
            if res[1] <= 0.05:
                better = "\\redcross"
                symbols.append('  ✘')
            else:
                better = "="
                symbols.append("")
        print(f"{wins} & {equal} & {loses} & {better} & {statistics.mean(m1valus)} & {statistics.mean(m2valus)}")
        methods.append(vers[0])
        percentages.append((100*wins/len(m1valus),100*equal/len(m1valus),100*loses/len(m1valus)))
    plot_percentages(percentages, methods, symbols,ax,title,colors)


def make_evol_not_evol():
    df = pd.read_csv("results.csv")
    df['instance'] = df['method_name'] + df['parameters_names'] + df['parameters_value']
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 3))
    colors = ['#34a0e2', '#eccda4', '#eae6df']
    Burst_Impact(df,axes[0],title="Not-Evolving datasets",dtype="notevol",colors=colors)
    Burst_Impact(df,axes[1],title="Evolving datasets",dtype="evol",colors=colors)

    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color=colors[0], label='Wins')
    blue_patch = mpatches.Patch(color=colors[1], label='Equal')
    orange_patch = mpatches.Patch(color=colors[2], label='Worse')

    handles = [
        green_patch,
        blue_patch,
        orange_patch,
        plt.Line2D([0], [0], color="white", label="✔ statistically significant better"),
    ]

    axes[0].legend(
        handles=handles,
        loc='upper center',
        ncols=4,
        fontsize=14,
        bbox_to_anchor=(1, 1.3)
    )
    # Burst_Impact(df,dtype="all")
    plt.show()

def make_all_all():
    df = pd.read_csv("results.csv")
    df['instance'] = df['method_name'] + df['parameters_names'] + df['parameters_value']
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 3))
    colors = ['#34a0e2', '#eccda4', '#eae6df']
    Burst_Impact(df,axes[0],title="ARI all datasets",dtype="all",metric='ARI',colors=colors)
    Burst_Impact(df,axes[1],title="Purity all datasets",dtype="all",metric='Purity',colors=colors)

    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color=colors[0], label='Wins')
    blue_patch = mpatches.Patch(color=colors[1], label='Equal')
    orange_patch = mpatches.Patch(color=colors[2], label='Worse')

    handles = [
        green_patch,
        blue_patch,
        orange_patch,
        plt.Line2D([0], [0], color="white", label="✔ statistically significant better"),
    ]

    axes[0].legend(
        handles=handles,
        loc='upper center',
        ncols=4,
        fontsize=14,
        bbox_to_anchor=(1, 1.3)
    )
    # Burst_Impact(df,dtype="all")
    plt.show()




def B_strategy_vs_no_K(metric='ARI'):
    plt.rcParams.update({
        'font.family': 'serif',  # Set font family to serif (you can use 'sans-serif', 'monospace', etc.)
        'font.size': 12,
        'font.weight': 'bold',  # Set the global font weight to bold
        # Set the global font size
    })
    temp_no_k_methods = ["SCKmedoids_SBD","SCKmedoids", "SCKmeans", "DbStream", "DenStream", "BirchN", "BURST"]
    build_df = {}
    df = pd.read_csv('results.csv')
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                "parameters_names", "parameters_value"], keep='last')
    dictdata = {}
    for vers in temp_no_k_methods:
        m1valus=[]
        for dataset in df["dataset_name"].unique():
            for ktype in ["init"]:
                dfdata = df[df["dataset_name"] == dataset]
                m1 = find_instance(dfdata, vers, ktype)[metric]
                m1valus.append(m1)

        dictdata[vers.replace("SC","B_").replace("_SBD","\n(SBD)    ")] = m1valus


    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10, 3))

    violinplot2(pd.DataFrame(dictdata), metric, axes[0][0])
    axes[0]=rankmodelspersetting(dictdata, f"Critical Diagram for {metric}",axes[0][1])

    metric="Purity"
    dictdata = {}
    for vers in temp_no_k_methods:
        m1valus = []
        for dataset in df["dataset_name"].unique():
            for ktype in ["init"]:
                dfdata = df[df["dataset_name"] == dataset]
                m1 = find_instance(dfdata, vers, ktype)[metric]
                m1valus.append(m1)

        dictdata[vers.replace("SC", "B_").replace("_SBD", "\n(SBD)     ")] = m1valus

    violinplot2(pd.DataFrame(dictdata), metric, axes[1][0])
    axes[0] = rankmodelspersetting(dictdata, f"Critical Diagram for {metric}", axes[1][1])

    plt.show()


def investigate_variants_get_best(df,method_name,metric):
    """
    df: results in df

    """
    all_varians=[f"{method_name}{variant}" for variant in ["_SBD","_22",""]]
    meds=[]
    values=[]
    tvalues=[]
    uns=len(df["dataset_name"].unique())
    for vers in all_varians:
        m1valus = []
        times = []
        for dataset in df["dataset_name"].unique():
            for ktype in ["init"]:
                dfdata = df[df["dataset_name"] == dataset]
                m1 = find_instance(dfdata, vers, ktype)
                if m1 is None:
                    continue
                m1m=m1[metric]
                m1valus.append(m1m)
                times.append((m1["Runtime"],m1["Fit_size"]*10))
        if len(m1valus)<0.9*uns:
            print(f"{vers} has {len(m1valus)}!=125 variants")
            meanv1 = 0
            medianv1 = -1
            meds.append(medianv1)
            values.append([-1,-1])
            tvalues.append([-1,-1])
            continue
        meanv1 = statistics.mean(m1valus)
        medianv1 = statistics.median(m1valus)
        meds.append(medianv1)
        values.append(m1valus)
        tvalues.append(times)
    posmax=meds.index(max(meds))
    return all_varians[posmax],values[posmax],tvalues[posmax]


def Allmethots(temp_no_k_methods,axes,dictdataall,allnames,initnames,colordict_f,metric='ARI'):
    plt.rcParams.update({
        'font.family': 'serif',  # Set font family to serif (you can use 'sans-serif', 'monospace', etc.)
        'font.size': 12,
        'font.weight': 'bold',  # Set the global font weight to bold
        # Set the global font size
    })

    dictdata = {}
    for vers in temp_no_k_methods:
        namen=allnames[initnames.index(vers)]
        dictdata[namen] = dictdataall[namen]
    violinplot(pd.DataFrame(dictdata), metric, axes,colordict_f=colordict_f)

def get_res_inner(all_methods,metric = "ARI", dataset_type="all"):
    dictdata={}
    df = pd.read_csv('results.csv')
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                "parameters_names", "parameters_value"], keep='last')
    if dataset_type == "evolv":
        df = df[df['number_of_classes_in_fit'] != df['number_of_classes_in_total']]
    elif dataset_type == "not_evolv":
        df = df[df['number_of_classes_in_fit'] == df['number_of_classes_in_total']]
    allnames=[]
    initnames=[]
    meds=[]
    for vers in all_methods:
        name, m1valus,_ = investigate_variants_get_best(df, vers, metric)
        name_n = name.replace("SC", "B_"). \
            replace("Static_22", "\ncatch22 (Sta@tic)").replace("Static_SBD", "\nSBD (S@tatic)"). \
            replace("_SBD", "\nSBD").replace("Static", "\n(Sta@tic)"). \
            replace("_22", "\ncatch22").replace("@", "")
        dictdata[name_n] = m1valus
        allnames.append(name_n)
        initnames.append(vers)
        meds.append(statistics.median(m1valus))
    sorted_indices = np.argsort(meds)
    allnames = [allnames[i] for i in sorted_indices]
    initnames = [initnames[i] for i in sorted_indices]
    return allnames,initnames,dictdata

def get_res_inner_Time(all_methods,metric = "ARI", dataset_type="all"):
    dictdata={}
    df = pd.read_csv('results.csv')
    df = df.drop_duplicates(
        subset=["dataset_name", "method_name", "Fit_size", "number_of_classes_in_fit", "number_of_classes_in_total",
                "parameters_names", "parameters_value"], keep='last')
    if dataset_type == "evolv":
        df = df[df['number_of_classes_in_fit'] != df['number_of_classes_in_total']]
    elif dataset_type == "not_evolv":
        df = df[df['number_of_classes_in_fit'] == df['number_of_classes_in_total']]
    allnames=[]
    initnames=[]
    meds=[]
    for vers in all_methods:
        name, m1valus,m1times = investigate_variants_get_best(df, vers, metric)
        name_n = name.replace("SC", "B_"). \
            replace("Static_22", "\ncatch22 (Sta@tic)").replace("Static_SBD", "\nSBD (S@tatic)"). \
            replace("_SBD", "\nSBD").replace("Static", "\n(Sta@tic)"). \
            replace("_22", "\ncatch22").replace("@", "")
        dictdata[name_n] = m1times
        allnames.append(name_n)
        initnames.append(vers)
        meds.append(statistics.median(m1valus))
    sorted_indices = np.argsort(meds)
    allnames = [allnames[i] for i in sorted_indices]
    initnames = [initnames[i] for i in sorted_indices]
    return allnames,initnames,dictdata

def new_summary_plot():
    temp_methods =["KShapesStatic","KShapes","Kmedoids","Kmeans","Clustream","StreamLS","DTC","BURSTK","BirchK","MBKmeans"]
    No_K_techniques = ["DbStream", "BURST", "DenStream", "BirchN","KShapesStatic"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))



    all_methods = ["DbStream", "BURST", "DenStream", "BirchN", "KShapesStatic", "KShapes", "Kmedoids", "Kmeans",
                   "Clustream", "StreamLS", "DTC", "BURSTK", "BirchK","MBKmeans"]

    allnames,initnames,dictdata=get_res_inner(all_methods, metric="ARI")
    color_list = sns.color_palette("coolwarm", len(allnames)).as_hex()
    colors={method: color for method, color in zip(initnames, color_list)}

    colordict_f = {method: colors[ininame] for method, ininame in zip(allnames, initnames)}
    allnames2,initnames2,dictdata2=get_res_inner(all_methods, metric="Purity")
    colordict_f2 = {method: colors[ininame] for method, ininame in zip(allnames2, initnames2)}

    axes[0][0].set_title("KC methods")
    axes[0][1].set_title("AC methods")
    Allmethots(temp_methods,axes[0][0],dictdata,allnames,initnames,colordict_f,metric='ARI')
    Allmethots(temp_methods, axes[1][0], dictdata2, allnames2, initnames2, colordict_f2, metric='Purity')


    Allmethots(No_K_techniques, axes[0][1], dictdata, allnames, initnames, colordict_f, metric='ARI')
    Allmethots(No_K_techniques, axes[1][1], dictdata2, allnames2, initnames2, colordict_f2, metric='Purity')
    plt.show()

def rankmodelspersetting_select(dictdataall,allnames,initnames,methods,ax):
    dictdata={}
    for vers in methods:
        namen=allnames[initnames.index(vers)]
        dictdata[namen] = dictdataall[namen]
    #matplotlib.rc('font', **font)
    dfdict=dictdata
    # for key in dictdata.keys():
    #     dfdict[key.replace("\n","_")]=dictdata[key]
    df2 = pd.DataFrame(dfdict)
    print(df2.shape[0])
    results = autorank.autorank(df2,
                                alpha=0.05,
                                verbose=False,
                                order='descending',
                                #force_mode='nonparametric'
                                )
    autorank.create_report(results)
    ax=autorank.plot_stats(results,width=4,ax=ax)

    return ax

def critical_diagrams_from_best_on_evolving():
    plt.rcParams.update({
        'font.family': 'serif',  # Set font family to serif (you can use 'sans-serif', 'monospace', etc.)
        'font.size': 14,
        'font.weight': 'bold',  # Set the global font weight to bold
        # Set the global font size
    })

    temp_methods = ["KShapesStatic", "KShapes", "Kmedoids", "Kmeans", "Clustream", "StreamLS", "DTC", "BURSTK",
                    "BirchK","MBKmeans"]
    No_K_techniques = ["DbStream", "BURST", "DenStream", "BirchN", "KShapesStatic"]

    all_methods = ["DbStream", "BURST", "DenStream", "BirchN", "KShapesStatic", "KShapes", "Kmedoids", "Kmeans",
                   "Clustream", "StreamLS", "DTC", "BURSTK", "BirchK","MBKmeans"]

    allnames, initnames, dictdata = get_res_inner(all_methods, metric="ARI",dataset_type="evolv")
    allnames2, initnames2, dictdata2 = get_res_inner(all_methods, metric="ARI", dataset_type="not_evolv")


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].set_title("AC on Evolving time-series")
    rankmodelspersetting_select(dictdata,allnames,initnames,No_K_techniques,axes[0])
    axes[1].set_title("AC on Not-Evolving time-series")
    rankmodelspersetting_select(dictdata2, allnames2, initnames2, No_K_techniques, axes[1])
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].set_title("KC on Evolving time-series")
    rankmodelspersetting_select(dictdata, allnames, initnames, temp_methods, axes[0])
    axes[1].set_title("KC on Not-Evolving time-series")
    rankmodelspersetting_select(dictdata2, allnames2, initnames2,temp_methods,axes[1])

    plt.show()

def time_plot():
    temp_methods =["KShapesStatic","KShapes","Kmedoids","Kmeans","Clustream","StreamLS","DTC","BURSTK","BirchK","MBKmeans"]
    No_K_techniques = ["DbStream", "BURST", "DenStream", "BirchN","KShapesStatic"]

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))



    all_methods = ["DbStream", "BURST", "DenStream", "BirchN", "KShapesStatic", "KShapes", "Kmedoids", "Kmeans",
                   "Clustream", "StreamLS", "DTC", "BURSTK", "BirchK","MBKmeans","SCKmeans"]

    allnames,initnames,dictdata=get_res_inner_Time(all_methods, metric="ARI")
    color_list = sns.color_palette("coolwarm", len(allnames)).as_hex()
    colors={method: color for method, color in zip(initnames, color_list)}

    means=[]
    for name in allnames:
        if "B_Kmeans" in name:
            tempname="B_Kmeans"
        else:
            tempname=name
        x=[t[1] for t in dictdata[tempname]]
        y=[t[0] for t in dictdata[tempname]]
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        x_sorted = list(x_sorted)
        y_sorted = list(y_sorted)
        means.append(np.mean(y_sorted))
    plt.bar(allnames, means, align='center')
    plt.show()

# critical_diagrams_from_best_on_evolving()

# 5
# new_summary_plot()
# time_plot()
# 5.3 plots
# make_evol_not_evol()
# make_all_all()
B_strategy_vs_no_K(metric='ARI')
# B_strategy_vs_no_K(metric='Purity')