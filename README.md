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
:exclamation: _If your project yielded or intends to yield some novel analysis, please include them in your readme. It can be named something other than results as well._ :exclamation:

## Team Members

Pradeep Varathan | ppugale@iu.edu | Team Leader    
Wei-An Chen |  | Member.  
Ghadah Alshaharani|  | Member.  
Mehmet Enes Inam | mehmet.enes.inam@gmail.com | Member.  
Karolina Willicott | kwillicott@crimson.ua.edu | Member.
Karen Bonilla| kabonill@iu.edu | Member    
Zaid Soomro, MD|  | Member.   
