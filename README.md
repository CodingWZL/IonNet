# AI for Solid Electrolyte Design
This software package implements the IonNet that takes compositional information to predict the ionic conductivity at room temperature.

The package provides three major functions:
* Calculate the multimodal descriptors based on chemcial compositions.
* Train a IonNet model with the entire data (including from scratch training and transfer learning).
* Generated substituted compounds and optimized the structures with MLPs.
* Predict conductivity based on IonNet model.
<img width="1232" height="524" alt="image" src="https://github.com/user-attachments/assets/7d7ea0f9-83f1-48ca-8ded-715c74f3096c" />


## Prerequisites
* torch==2.2.1
* rdkit==2024.3.6
* scikit-learn==1.5.1
* matminer==0.9.2
* pymatgen==2024.9.17


The easiest way of installing the prerequisites is via `conda`. After installing `conda`, run the following command to create a new environment named `ionnet` and install all prerequisites:

    conda upgrade conda
    conda create -n ionnet python=3.12 scikit-learn pytorch rdkit pysr

This creates a `conda` environment for running `IonNet`. Before using `IonNet`, activate the environment by:
    
    source activate ionnet

Alternatively, `environment.yaml` provides the dependencies for creating running environment. Then, in directory `model`, you can test if all the prerequisites are installed properly by running:

    python train.py

After you finished using `IonNet`, exit the environment by:

    source deactivate

## Data
We provide the datasets in the `data` folder, including three datasets:

* Computational data: `Li-IonML-Computations.csv`, which contains 8,950 computational samples for model training.
* Experimental data: `LiIonDatabase-Experiments-300K.csv`, which contains 398 experimental samples for transfer learning.
* Prediction data: `Li-MP-final.csv`, which 4,583 compounds from Materials Project.

## Descriptor
In this work, three types of representations were used to build the model: meredig, magpis, and megnet. Meredig descriptor is a 120-dimensional vector36, atomic fraction of each of the first 103 elements, in order of atomic number, and 17 statistics of elemental properties: mean atomic weight of constituent elements. Magpie, the materials agnostic platform for informatics and exploration, is a versatile tool designed to streamline the development of the ML models from materials data. Megnet, which represents the non-linear element embeddings generated using the materials graph network.

We provide the script for calculating the represetations, one can use:

    python composition_feature.py

More, the calculated descriptor data is provided in the `descriptor` folder.

* Descriptors for computational data: meredig, magpie, and megnet.
* Descriptors for experimental data: meredig, magpie, and megnet.

## Model
We provide the scripts for model training and finu-tuning. During the model training, the data is loaded by `dataload.py`. The training in from scratch mode on the computational data can be implemented by running:

    python train.py # change the computation=True in line 54

The transfer learning training on the experimental data can be implemented by running:

    python fine_tune.py # # change the computation=Flase in line 112

Using the well-trained models in `trained_models`, one can predict the ionic conductivity with the descriptors by using:

    python predict.py

## Substitution
A script for creating the substituted compounds from the screened Materials Project (`Li-MP-final.csv`) is provided. Two functions, including the single-element substitution and double-element substitution are given:

    python substitute.py

A script for generating possible substituted structures based on the original structure is provided. The structures will be saved as CIF files, and then the MLPs can be used to evaluate the formation energy or to optimize the structures directly.

    python generation.py

## Author contributions
This software was primarily written by `Dr. Zhilong Wang` who is advised by `Prof. Fengqi You`.

## How to cite
Please cite the following work if you want to use IonNet:

    Zhilong Wang, Fengqi You*. Submitted (2025). 20/09/2025


