import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_data(computation):
    if computation is False:
        # Load the features
        feature_1 = torch.from_numpy(np.load("experiment_meredig_feature.npy")[16:, :]).float() # 120
        feature_2 = torch.from_numpy(np.load("experiment_magpie_feature.npy")[16:, :]).float() # 132
        feature_3 = torch.from_numpy(np.load("experiment_megnet_feature.npy")[16:, :]).float() # 80
        target = torch.from_numpy(np.loadtxt("../../data/experiment-target.txt")[16:]).float().unsqueeze(1)  # target
        batch_size = 32
    else:
        feature_1 = torch.from_numpy(np.load("BVSE-computation_meredig_feature.npy")).float()  # 120
        feature_2 = torch.from_numpy(np.load("BVSE-computation_magpie_feature.npy")).float()  # 132
        feature_3 = torch.from_numpy(np.load("BVSE-computation_megnet_feature.npy")).float()  # 80
        target = torch.from_numpy(np.loadtxt("../../data/BVSE-target.txt")).float().unsqueeze(1)  # target
        batch_size = 256


    # Create dataset
    dataset = TensorDataset(feature_1, feature_2, feature_3, target)

    # Split dataset into training, validation, test sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    #test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for batch training
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # return train_loader, val_loader
    return dataset
