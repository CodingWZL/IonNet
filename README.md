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

(1) Computational data: `Li-IonML-Computations.csv`, which contains 8,950 computational samples for model training.

(2) Experimental data: `LiIonDatabase-Experiments-300K.csv`, which contains 398 experimental samples for transfer learning.

(3) Prediction data: `Li-MP-final.csv`, which 4,583 compounds from Materials Project.

## Descriptor
In this work, three types of representations were used to build the model: meredig, magpis, and megnet. Meredig descriptor is a 120-dimensional vector36, atomic fraction of each of the first 103 elements, in order of atomic number, and 17 statistics of elemental properties: mean atomic weight of constituent elements. Magpie, the materials agnostic platform for informatics and exploration, is a versatile tool designed to streamline the development of the ML models from materials data. Megnet, which represents the non-linear element embeddings generated using the materials graph network.

We provide the script for calculating the represetations, one can use:

    python composition_feature.py

More, the calculated descriptor data is provided in the `descriptor` folder.

(1) Descriptors for computational data: meredig, magpie, and megnet.

(2) Descriptors for experimental data: meredig, magpie, and megnet.


