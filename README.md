<h1 align="center">BURST Clustering</h1>

<hr>

Batch-Updating Real-time Assignment STrategy (BURST),
a general framework for enabling real-time time-series clustering.
BURST extends static clustering methods to perform real-time assignments
while incorporating batch updates to handle evolving time-series data,
allowing clusters to merge, emerge, or disappear over time. 
Finally, AutoKC algorithm is leveraged by BURST to 
eliminate the need for prior knowledge of the number of existing clusters.

![BURST outline](/images/BURST.png)


## Running clustering

Ensure you have Conda installed. You can install Miniconda or Anaconda from [here](https://docs.anaconda.com/miniconda/install/):
```bash
conda --version
```

Create conda environment from `enviorment.yml`:
```bash
conda env create -f environment.yml
```

Activate conda environment:
```bash
conda activate Profiling
```

### Running a clustering experiment
To run the clustering algorithm on specific dataset (e.g. `UCR_SyntheticControl`), use the following command:
```bash
python run.py BURST "UCR_SyntheticControl"
```
Output:
```
{'Purity': 0.6537037037037037, 'RI': np.float64(0.8143475572047001), 'ARI': 0.38849841200051877, 'NMI': np.float64(0.5814636474577509), 'Runtime': 17.169285535812378}
```

### Pass Parameters
To modify the default parameters of the algorithms, use the --modified_params argument and pass a JSON-like string in the following format:
```
{ "parameter_name": value }
```
Default parameters are listed in file allMethods.py.

Example: Modify distance measure for Kmedoids algorithm
```bash
python run.py Kmedoids "UCR_SyntheticControl" --modified_params '{"distance_measure":"euclidean"}'
```
Output:
```
{'Purity': 0.18888888888888888, 'RI': np.float64(0.5981859410430839), 'ARI': 0.19889250616589238, 'NMI': np.float64(0.270411863694917), 'Runtime': 0.011178255081176758}
```

Setting distance metric to SBD:
```bash
python run.py Kmedoids "UCR_SyntheticControl" --modified_params '{"distance_measure":"euclidean"}'
Output:
```
Output:
```
{'Purity': 0.21481481481481482, 'RI': np.float64(0.5557273414416272), 'ARI': 0.12776395730559226, 'NMI': np.float64(0.18425190228843724), 'Runtime': 0.3693063259124756}
```

### Getting Help
For more information on available commands and arguments, you can run:
```bash
python run.py --help
```
## Data

For the experimental analyses [UCR Time Series Datasets](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) archive was used:

Dataset folders should be placed under UCR_DATASETS folder as the example of SyntheticControl.


## AutoKC capabilities

This repository contains the AutoKC algorithm, which is a clustering algorithm that can automatically determine the number of clusters in a dataset for partition-clustering methods:

![AutoKC](/images/AutoKC.png)

Example of AutoKC on baguette-shaped clusters (source: Examples/Baguette_shape_clustering/check_autoKC_clustering.py):
![AutoKC](/images/AutoKC_baguette.png)

[Redirect here for BURST effectiveness evaluation plots.](plots_and_results/README.md)
## Acknowledgment

### Repositories:
TSB-UAD: https://github.com/TheDatumOrg/TSB-UAD
