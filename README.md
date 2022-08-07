# Using Graph Neural Network to decrypt Personalized Medicare in Alzheimer's Disease

## Table of Contents

- [Template](#team-repo-template)
    - [Background](#Background)
    - [Data](#data)
    - [Usage](#usage)
        - [Installation](#installation)
        - [Requirements](#requirements) _Can be named Dependencies as well_
        - [Activate conda environment](#activate-conda-environment) _Optional_
        - [Steps to run ](#steps-to-run) _Optional depending on project_
            - [Step-1](#step-1)
            - [Step-2](#step-2)
    - [Results](#results) _Optional depending on project_
    - [Team Members](#team-members)

## Background

Our goal is to capitalize on Graph Layer Relevance Propogation method which explores a Graph Convolutional Neural Network, to decode the pathological or node level difference between Alzheimer Disease subjects and control patients. 

## Introduction

Alzheimer’s disease (AD) is the most common form of dementia (60-70%) mainly affecting the elderly (age >65) with an estimated annual cost of about $300 billion USD (“2020 Alzheimer’s Disease Facts and Figures,” 2020; Dementia, n.d.; Winston Wong, 2020).  return  
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

We are planning to build this whole pipeline into the docker image for easy installation and running. It would be as simple as providing the expression set file, PPI network file and hyperparameters in a config file or could also have an option of auto tuning using tools inside the container such as Optuna.

### Installation

:exclamation: _If installation is required, please mention how to do so here._ :exclamation:

Installation simply requires fetching the source code. Following are required:

- Git

To fetch source code, change in to directory of your choice and run:

```sh
git clone -b main \
    git@github.com:u-brite/team-repo-template.git
```

### Requirements



*OS:*

Currently works only in Linux OS. Docker versions may need to be explored later to make it useable in Mac (and
potentially Windows).

*Tools:*

- Anaconda3
    - Tested with version: 2020.02

### Activate conda environment
:exclamation: _Optional: Depends on project._ :exclamation:

Change in to root directory and run the commands below:

```sh
# create conda environment. Needed only the first time.
conda env create --file configs/environment.yaml

# if you need to update existing environment
conda env update --file configs/environment.yaml

# activate conda environment
conda activate testing
```

### Steps to run
:exclamation: _Optional: Depends on project._ :exclamation:

#### Step 1

```sh
python src/data_prep.py -i path/to/file.tsv -O path/to/output_directory
```

#### Step 2

```sh
python src/model.py -i path/to/parsed_file.tsv -O path/to/output_directory
```

Output from this step includes -

```directory
output_directory/
├── parsed_file.tsv               <--- used for model
├── plot.pdf- Plot to visualize data
└── columns.csv - columns before and after filtering step

```

**Note**: The is an example note with a [link](https://github.com/u-brite/team-repo-template).


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

Pradeep Varathan | ppugale@iu.edu | Team Leader    
Wei-An Chen | wchen678@gatech.edu | Member.  
Ghadah Alshaharani|  | Member.  
Mehmet Enes Inam | mehmet.enes.inam@gmail.com | Member.  
Karolina Willicott | kwillicott@crimson.ua.edu | Member.  
Karen Bonilla| kabonill@iu.edu | Member    
Zaid Soomro, MD|  | Member.   
