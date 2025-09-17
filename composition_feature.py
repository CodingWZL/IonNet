import cgi
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.composition import Meredig, ElementProperty
from pymatgen.core import Composition
import pandas as pd
import numpy as np
import multiprocessing


class ComponentFeatures():
    # calculate the atomic fraction of each element in a compostion from a "Formula" file, dimension: 103
    def element_fraction_feature(self, formula):
        element_fraction = ElementFraction()
        composition = Composition(formula)
        feature = element_fraction.featurize(composition)
        # file.write(str(element_fraction).replace(","," ").replace("[","").replace("]","") + "\n")
        return feature

    def meredig_element_feature(self, formula):
        """
        Features:
            Atomic fraction of each of the first 103 elements, in order of atomic number.
            17 statistics of elemental properties;
                Mean atomic weight of constituent elements (1)
                Mean periodic table row and column number (2)
                Mean and range of atomic number (2)
                Mean and range of atomic radius (2)
                Mean and range of electronegativity (2)
                Mean number of valence electrons in each orbital (4)
                Fraction of total valence electrons in each orbital (4)

        """
        meredig = Meredig()
        composition = Composition(formula)
        feature = meredig.featurize(composition)
        # file.write(str(features).replace(","," ").replace("[","").replace("]","") + "\n")
        return feature

    def magpie_element_feature(self, formula):
        # "Number",: Atomic number
        # "MendeleevNumber",
        # "AtomicWeight",: Atomic weight
        # "MeltingT",:  Melting temperature of element
        # "Column",: Column on periodic table
        # "Row",: Row on periodic table
        # "CovalentRadius",: Covalent radius of each element
        # "Electronegativity",: Pauling electronegativity
        # "NsValence",:  Number of filled s valence orbitals
        # "NpValence",: Number of filled p valence orbitals
        # "NdValence",
        # "NfValence",
        # "NValence",: Number of valence electrons
        # "NsUnfilled",: Number of unfilled s valence orbitals
        # "NpUnfilled",
        # "NdUnfilled",
        # "NfUnfilled",
        # "NUnfilled",: Number of unfilled valence orbitals
        # "GSvolume_pa",: DFT volume per atom of T=0K ground state
        # "GSbandgap",: DFT bandgap energy of T=0K ground state
        # "GSmagmom",: DFT magnetic momenet of T=0K ground state
        # "SpaceGroupNumber": Space group of T=0K ground state structure
        # "minimum", "maximum", "range", "mean", "avg_dev", "mode"
        element_property = ElementProperty.from_preset("magpie")
        # element_property = ElementProperty.from_preset("matminer")
        composition = Composition(formula)
        feature = element_property.featurize(composition)
        return feature

    def matminer_element_feature(self, formula):
        # calculate elemental property attributes
        # "X",
        # "row",
        # "group",
        # "block",
        # "atomic_mass",
        # "atomic_radius",
        # "mendeleev_no",
        # "electrical_resistivity",
        # "velocity_of_sound",
        # "thermal_conductivity",
        # "melting_point",
        # "bulk_modulus",
        # "coefficient_of_linear_thermal_expansion",
        # "minimum", "maximum", "range", "mean", "std_dev"
        element_property = ElementProperty.from_preset("matminer")
        composition = Composition(formula)
        feature = element_property.featurize(composition)
        return feature

    def deml_element_feature(self, formula):
        # calculate elemental property attributes
        # "atom_num",
        # "atom_mass",
        # "row_num",
        # "col_num",
        # "atom_radius",
        # "molar_vol",
        # "heat_fusion",
        # "melting_point",
        # "boiling_point",
        # "heat_cap",
        # "first_ioniz",
        # "electronegativity",
        # "electric_pol",
        # "GGAU_Etot",
        # "mus_fere",
        # "FERE correction",
        # "minimum", "maximum", "range", "mean", "std_dev"
        element_property = ElementProperty.from_preset("deml")
        composition = Composition(formula)
        feature = element_property.featurize(composition)
        return feature

    def megnet_element_feature(self, formula):
        # calculate elemental property attributes
        # "minimum", "maximum", "range", "mean", "std_dev"
        element_property = ElementProperty.from_preset("megnet_el")
        composition = Composition(formula)
        feature = element_property.featurize(composition)
        return feature

    def matscholar_element_feature(self, formula):
        # calculate elemental property attributes
        # "minimum", "maximum", "range", "mean", "std_dev"
        element_property = ElementProperty.from_preset("matscholar_el")
        composition = Composition(formula)
        feature = element_property.featurize(composition)
        return feature


######################
# Experimental feature
######################
def experiment_data():
    experimental_data = pd.read_csv("data/LiIonDatabase-Experiments-300K.csv")
    experimental_formula = experimental_data['composition'].tolist()
    print(len(experimental_formula))

    with multiprocessing.Pool(processes=3) as pool:
        feature_1 = pool.apply_async(calculate_meredig, args=(experimental_formula,))
        feature_2 = pool.apply_async(calculate_magpie, args=(experimental_formula,))
        feature_3 = pool.apply_async(calculate_megnet, args=(experimental_formula,))

        experiment_meredig_feature = feature_1.get()
        experiment_magpie_feature = feature_2.get()
        experiment_megnet_feature = feature_3.get()

    np.save("experiment_meredig_feature.npy", experiment_meredig_feature)  # dimension = 120
    np.save("experiment_magpie_feature.npy", experiment_magpie_feature)  # dimension = 132
    np.save("experiment_megnet_feature.npy", experiment_megnet_feature)  # dimension = 80


def calculate_meredig(formula):
    cf = ComponentFeatures()
    meredig_feature = []
    for i in formula:
        # print(i)
        meredig_feature.append(cf.meredig_element_feature(i))
    return np.array(meredig_feature)

def calculate_magpie(formula):
    cf = ComponentFeatures()
    magpie_feature = []
    for i in formula:
        magpie_feature.append(cf.magpie_element_feature(i))
    return np.array(magpie_feature)

def calculate_megnet(formula):
    cf = ComponentFeatures()
    megnet_feature = []
    for i in formula:
        megnet_feature.append(cf.megnet_element_feature(i))
    return np.array(megnet_feature)



#######################
# Computational feature
#######################
def computation_data():
    computational_data = pd.read_csv("../../data/Li-IonML-Computations.csv")
    computational_formula = computational_data['formula'].tolist()
    print(len(computational_formula))

    with multiprocessing.Pool(processes=3) as pool:
        feature_1 = pool.apply_async(calculate_meredig, args=(computational_formula,))
        feature_2 = pool.apply_async(calculate_magpie, args=(computational_formula,))
        feature_3 = pool.apply_async(calculate_megnet, args=(computational_formula,))

        computation_meredig_feature = feature_1.get()
        computation_magpie_feature = feature_2.get()
        computation_megnet_feature = feature_3.get()


    np.save("BVSE-computation_meredig_feature.npy", computation_meredig_feature)  # dimension = 120
    np.save("BVSE-computation_magpie_feature.npy", computation_magpie_feature)  # dimension = 132
    np.save("BVSE-computation_megnet_feature.npy", computation_megnet_feature)  # dimension = 80




#######################
# Prediction feature
#######################
def prediction_data():
    prediction_data = pd.read_csv("Li-MP-final.csv")
    prediction_formula = prediction_data['Formula'].tolist()
    print(len(prediction_formula))

    with multiprocessing.Pool(processes=3) as pool:
        feature_1 = pool.apply_async(calculate_meredig, args=(prediction_formula,))
        feature_2 = pool.apply_async(calculate_magpie, args=(prediction_formula,))
        feature_3 = pool.apply_async(calculate_megnet, args=(prediction_formula,))

        prediction_meredig_feature = feature_1.get()
        prediction_magpie_feature = feature_2.get()
        prediction_megnet_feature = feature_3.get()


    np.save("mp_meredig_feature.npy", prediction_meredig_feature)  # dimension = 120
    np.save("mp_magpie_feature.npy", prediction_magpie_feature)  # dimension = 132
    np.save("mp_megnet_feature.npy", prediction_megnet_feature)  # dimension = 80



if __name__ == "__main__":
    # experiment_data()
    # computation_data()
    prediction_data()

    # data = np.load("computation_magpie_feature.npy")
    # print(data[0:10])
    # data = np.isnan(np.load("computation_meredig_feature.npy"))
    # rows_to_remove = [5854]
    #
    # # Create a boolean mask to keep the rows that are not in rows_to_remove
    # mask = np.ones(data.shape[0], dtype=bool)
    # mask[rows_to_remove] = False
    #
    # # Use the mask to filter out the rows
    # filtered_data = data[mask]
    #
    # # Save the modified array back to a new .npy file
    # np.save('computation_meredig_feature.npy', filtered_data)

    # print(data)



