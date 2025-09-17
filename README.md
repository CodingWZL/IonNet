

# Descriptor
In this work, three types of representations were used to build the model: meredig, magpis, and megnet. Meredig descriptor is a 120-dimensional vector36, atomic fraction of each of the first 103 elements, in order of atomic number, and 17 statistics of elemental properties: mean atomic weight of constituent elements. Magpie, the materials agnostic platform for informatics and exploration, is a versatile tool designed to streamline the development of the ML models from materials data. Megnet, which represents the non-linear element embeddings generated using the materials graph network.

We provide the script for calculating the represetations, one can use:

    python composition_feature.py

More, the calculated descriptor data is provided in the `descriptor` folder.

(1) Descriptors for computational data: meredig, magpie, and megnet

(2) Descriptors for experimental data: meredig, magpie, and megnet


