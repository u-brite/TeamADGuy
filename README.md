# Using Graph Neural Network to decrypt Personalized Medicare in Alzheimer's Disease

## Table of Contents

- [Template](#team-repo-template)
    - [Background](#Background)
    - [Data](#data)
    - [Usage](#usage)
        - [Installation](#installation)
        - [Requirements](#requirements) 
        - [Activate conda environment](#activate-conda-environment) 
        - [Steps to run ](#steps-to-run)
            - [Step-1](#step-1)
            - [Step-2](#step-2)
    - [Results](#results) 
    - [Team Members](#team-members)

## Background

Our goal is to capitalize on Graph Layer Relevance Propogation method which explores a Graph Convolutional Neural Network, to decode the pathological or node level difference between Alzheimer Disease subjects and control patients. 

## Introduction

Alzheimer’s disease (AD) is the most common form of dementia (60-70%) mainly affecting the elderly (age >65) with an estimated annual cost of about $300 billion USD (“2020 Alzheimer’s Disease Facts and Figures,” 2020; Dementia, n.d.; Winston Wong, 2020). 

There is no cure for AD, and in the past twenty years only two drugs (Aducanumab and Gantenerumab) have had a potential to show clinically meaningful results (Commissioner, 2021; Gantenerumab | ALZFORUM, n.d.; How Is Alzheimer’s Disease Treated?, n.d.; Ostrowitzki et al., 2017; Tolar et al., 2020).
Exploration of additional biomarkers for this complex disease is, therefore, warranted and could potentially aid in the early detection or therapeutic intervention of AD patients.


## Methods 

We wish to develop a multiplex machine learning (ML) approach to identify [gene]omics biomarkers in AD and mild cognitive impairment (MCI) compared to healthy controls (HC). 
1. Identify best ML model that predicts AD or MCI versus HC
2. Apply this model on a validation set to confirm the performance
3. Combine multiple datasets to see if model performance improves


## Data

As for the deep learning model and relevance propagation method, we will follow the GCN Paper that has applied this method in the cancer biology filed with slight changes such as:
1. Expression Dataset from ROSMAP
2. PPI network from HPRD, or test other suitable network
3. Hyperparameter tuning


## Usage

We are planning to build this whole pipeline into python file with config for easy installation and running. It would be as simple as providing the expression set file, PPI network file and hyperparameters in a config file.

### Installation

Installation simply requires fetching the source code. Following are required:

- Git

To fetch source code, change in to directory of your choice and run:

```sh
git clone -b main \
    https://github.com/u-brite/TeamADGuy
```

### Requirements


*OS:*

Works in all available OS.

*Tools:*

- Anaconda3
    - Tested with version: 2020.02

### Activate conda environment

Change in to root directory and run the commands below to run the deep learning model:

```sh
# create conda environment. Needed only the first time.
conda env create --file configs/environment.yml

# if you need to update existing environment
conda env update --file configs/environment.yml

# activate conda environment
conda activate gcn
```

### Steps to run

#### Step 1
To run the deep learning model, the first step would include downloading your required file with expression dataset having subjects as columns and genes as rows while also a column under the name 'Probe' with the gene names for reference in the future. The final output data,which is the subject disease condition is also required for the prediction and finally, we would require the network, for which we used the HPRD PPI and you can freely use the PPI network that suits best.

The rough test files are present inside the Test folder for reference.

#### Step 2
Running the model requires completion of the config file with self explained headers present inside. Finally, the input of the python file would just be the config file itself.

```sh
python src/model.py -i path/to/parsed_file.tsv -O path/to/output_directory
```

Output from this step includes -

```directory
output_directory/
├── prediction.csv              
└── columns.csv - columns before and after filtering step

```



## Results

#### ROSMAP 

<p align = "center"> 

![ROSMAP Chart Results](/results/assets/ROSMAP_chart.jpg)

![ROSMAP Histogram](/results/assets/ROSMAP_Results.png)

</p>

#### GSE63063

<p align = "center"> 

![GSE63063 Chart Results](/results/assets/GSE63063_chart.jpg)

![GSE63063 Histogram](/results/assets/GSE63063_results.png)

</p>

#### miRNA

<p align = "center"> 

![miRNA Chart Results](/results/assets/miRNA_chart.jpg)

![miRNA Histogram](/results/assets/miRNA_results.png)

</p>

## Team Members

Pradeep Varathan | ppugale@iu.edu | Team Leader.  
Karen Bonilla| kabonill@iu.edu | Member.   
Mehmet Enes Inam | mehmet.enes.inam@gmail.com | Member.  
Karolina Willicott | kwillicott@crimson.ua.edu | Member.  

