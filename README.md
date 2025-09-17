# Data
We provide the datasets in the `data` folder, including three datasets:

(1) Computational data: `Li-IonML-Computations.csv`, which contains 8,950 computational samples for model training.

(2) Experimental data: `LiIonDatabase-Experiments-300K.csv`, which contains 398 experimental samples for transfer learning.

(3) Prediction data: `Li-MP-final.csv`, which 4,583 compounds from Materials Project.

# Descriptor
In this work, three types of representations were used to build the model: meredig, magpis, and megnet. Meredig descriptor is a 120-dimensional vector36, atomic fraction of each of the first 103 elements, in order of atomic number, and 17 statistics of elemental properties: mean atomic weight of constituent elements. Magpie, the materials agnostic platform for informatics and exploration, is a versatile tool designed to streamline the development of the ML models from materials data. Megnet, which represents the non-linear element embeddings generated using the materials graph network.

We provide the script for calculating the represetations, one can use:

    python composition_feature.py

More, the calculated descriptor data is provided in the `descriptor` folder.

(1) Descriptors for computational data: meredig, magpie, and megnet.

(2) Descriptors for experimental data: meredig, magpie, and megnet.


