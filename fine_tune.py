import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from dataload import load_data
from train import evaluate, print_results
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler


# load the trained model
def load_model(model, path='model-9.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# interface of transfer learning, freeze partial parameters of hidden layers
def freeze_partial_layers(model):
    # # freeze the first layer of attention network 1
    # for param in model.attn1.fc1.parameters():
    #     param.requires_grad = False
    # #
    # # # freeze the second layer of attention network 1
    # for param in model.attn1.fc2.parameters():
    #     param.requires_grad = False
    # #
    for param in model.attn1.fc3.parameters():
        param.requires_grad = False
    # #
    # # # freeze the first layer of attention network 2
    # for param in model.attn2.fc1.parameters():
    #     param.requires_grad = False
    # #
    # # # freeze the second layer of attention network 2
    # for param in model.attn2.fc2.parameters():
    #     param.requires_grad = False
    # # #
    for param in model.attn2.fc3.parameters():
        param.requires_grad = False

    # freeze the first layer of attention network 3
    # for param in model.attn3.fc1.parameters():
    #     param.requires_grad = False

    # freeze the second layer of attention network 3
    # for param in model.attn3.fc2.parameters():
    #     param.requires_grad = False
    #
    for param in model.attn3.fc3.parameters():
        param.requires_grad = False

    print("Frozen specific layers in both networks.")
    return model




# transfer learning: fine-tuning
def fine_tune_model(model, train_loader, val_loader):

    # freeze partial parameters
    model = freeze_partial_layers(model)

    # initialize the optimizer, and update the unfreeze parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0015)
    criterion = nn.MSELoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    # fine-tuning
    epochs = 300
    flag = 1000
    model_flag = model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0
        for batch_x1, batch_x2, batch_x3, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x1, batch_x2, batch_x3)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()
        for batch_x1, batch_x2, batch_x3, batch_y in val_loader:
            outputs = model(batch_x1, batch_x2, batch_x3)
            loss_val = criterion(outputs, batch_y)
            running_loss_val += loss_val.item()
        if running_loss_val / len(val_loader) < flag:
            flag = running_loss_val / len(val_loader)
            model_flag = model
            #print("The best model is saved")
        if (epoch + 1) % 10 == 0:
            print(f'Fine-tuning Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}')
            print(f'Validation Loss: {running_loss_val / len(val_loader)}')

    return model_flag


# save model
def save_model(model, path='fine_tuned_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":


    # data for transfer learning
    dataset = load_data(computation=False)

    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # fine-tune via transfer learning
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(fold)
        # load the well-trained model
        model = load_model(create_model(), path="experiment/fine_tuned_model8.pth")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)

        fine_tuned_model = fine_tune_model(model, train_loader, val_loader)
        save_model(fine_tuned_model, path="fine_tuned_model" + str(fold+1) + ".pth")

        # Get predictions and labels
        train_preds, train_labels = evaluate(fine_tuned_model, train_loader)
        val_preds, val_labels = evaluate(fine_tuned_model, val_loader)

        print_results(train_preds, train_labels, "Training" + str(fold+1))
        print_results(val_preds, val_labels, "Validation" + str(fold+1))

