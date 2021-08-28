# tascCODA_reproducibility
 Reproductibility repository for "tascCODA: Bayesian tree-aggregated analysis of compositional amplicon and single-cell data" (Ostner et al., 2021)).

This repository contains the data generation and analysis scripts used to produce the results in the manuscript,
as well as all figures generated in the process.

## Setup
To reproduce the results, first install the packages listed in the `requirements.txt` file. 

Then, download the raw and intermediate data from [zenodo](10.5281/zenodo.5302135) and unpack it in the same directory as this repository:

```
<parent_directory>
    |
    |_ tascCODA_reproducibility
    |
    |_ tascCODA_data
```

## Instructions

The repository is separated into three parts. 

### 1. Synthetic benchmarks
The `benchmarks` directory contains all simulations and analyses related to the synthetic data benchmarks.
Functions used for data generation and execution in all three benchmarks are contained in the `scripts` directory.

For all three simulation studies, the respective subdirectory contains: 
- One jupyter notebook for data generation
- One python script for running the benchmark
- One jupyter notebook for analysis of the produced results. 

The plots generated for each benchmark are stored in the subdirectory `plots`.
The model comparison benchmark additionally contains a jupyter notebook that evaluates the generated data with all models except scCODA and tascCODA.

**NOTE: The python scripts to run the benchmarks were executed on the HPC cluster of Helmholtz Zentrum München and will not work elsewhere! Running the benchmarks is also very resource intensive.**

### 2. Real data applications
The `applications` directory contains the scripts and plots used for the real data applications on single-cell (`smillie_UC`)
and microbial (`labus_IBS`) sequencing data.

**scRNA-seq data**

For the analysis of Ulcerative Colitis, three jupyter notebooks are available:

- 1_preprocessing_Smillie performs all preprocessing steps to prepare the data for analysis
- 2_Smillie_treeAgg_run applies tascCODA, scCDOA and Dirichlet regression to the different subsets of the data and generates all relating plots
- 3_Smillie_oos_prediction performs the out-of-sample prediction analysis

The plots generated are stored in the subdirectory `plots_smillie`.

**16S sequencing data**

The analysis of Irritable Bowel Syndrome is entirely performed in the jupyter notebook `tasccoda_analysis_Labus`.
`ibs_util_functions` contains some functions used by this notebook.

The plots generated are stored in the subdirectory `IBS_plots`.

### 3. Other material

This directory contains scripts, plots and other material for producing figures 1 and S1 from the paper.
