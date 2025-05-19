import pandas as pd
import autorank
from matplotlib import pyplot as plt


def rankmodelspersetting(dfdictini,ax):

    #matplotlib.rc('font', **font)
    dfdict={}
    for key in dfdictini.keys():
        dfdict[key.replace("\n","_")]=dfdictini[key]
    df2 = pd.DataFrame(dfdict)
    print(df2.head(30))
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


def read_mehtod(keepmethods,parameters=None):
    df = pd.read_csv('resultsUP.csv')
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

def get_low_high_datasets():
    dfshsiloute=pd.read_csv("shilooueteLabelsSBD")
    data_top_30 = dfshsiloute.nlargest(30, 'score')["dataset"].tolist()
    data_bottom_30 = dfshsiloute.nsmallest(30, 'score')["dataset"].tolist()
    return [f"UCR_{dn}" if "generate_" not in dn else dn  for dn in data_top_30 ], [f"UCR_{dn}" if "generate_" not in dn else dn  for dn in data_bottom_30]

def BURST_performance(data_top_30, data_bottom_30,metric="ARI",axlist=[]):
    No_K_techniques = ["DbStream", "BURST", "DenStream", "BirchN"]
    parameters = ["catch22", None,None,None]

    bottom_30_performance = {}
    top_30_performance = {}

    for method_name,parameters in zip(No_K_techniques,parameters):
        dfmethod = read_mehtod([method_name],parameters)
        dfmethod = dfmethod[dfmethod["dataset_name"].isin(data_bottom_30)]
        bottom_30_performance[method_name] = dfmethod[metric].values

        dfmethod = read_mehtod([method_name], parameters)
        dfmethod = dfmethod[dfmethod["dataset_name"].isin(data_top_30)]
        top_30_performance[method_name] = dfmethod[metric].values

    ###

    rankmodelspersetting(top_30_performance, axlist[0])
    rectangle_plot(axlist[2], top_30_performance["BirchN"], top_30_performance["BURST"])
    axlist[2].set_ylabel(f"BURST {metric}")
    axlist[2].set_xlabel(f"BirchN {metric}")

    rankmodelspersetting(bottom_30_performance, axlist[1])
    rectangle_plot(axlist[3], bottom_30_performance["BirchN"], bottom_30_performance["BURST"] )
    axlist[3].set_ylabel(f"BURST {metric}")
    axlist[3].set_xlabel(f"BirchN {metric}")

def rectangle_plot(ax,list1,list2):
    color = plt.cm.coolwarm(30)
    color2 = plt.cm.coolwarm(230)
    ax.scatter(list1, list2,marker="s",color=color)
    ax.plot([0,1],[0,1],color=color2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    return

def plot_shillouette_scores():
    data_top_30, data_bottom_30=get_low_high_datasets()
    fig, axes = plt.subplots(2, 4, figsize=(7, 3))
    axes[0][0].set_title("Top 30 silhouette score datasets")
    axes[0][2].set_title("Lowest 30 silhouette score datasets")

    BURST_performance(data_top_30, data_bottom_30,metric="ARI",axlist=[axes[0][0],axes[0][2],axes[0][1],axes[0][3]])
    BURST_performance(data_top_30, data_bottom_30,metric="Purity",axlist=[axes[1][0],axes[1][2],axes[1][1],axes[1][3]])
    fig.text(0.03, 0.75, "ARI", ha="center",fontsize=12,rotation=90)
    fig.text(0.03, 0.33, "Purity", ha="center",fontsize=12,rotation=90)

    plt.show()


plot_shillouette_scores()


