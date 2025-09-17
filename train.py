# train.py
import numpy as np
import torch
import torch.optim as optim
from model import create_model, create_optimizer, create_criterion
from dataload import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchmetrics.regression import PearsonCorrCoef


# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to the {path}")


# Evaluate on validation and test sets
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x1, batch_x2, batch_x3, batch_y in loader:
            outputs = model(batch_x1, batch_x2, batch_x3)
            all_preds.append(outputs)
            all_labels.append(batch_y)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return preds, labels


# Print results
def print_results(preds, labels, name):
    # automatically adjust the predictions that are not in 0-1
    # preds[preds < 0] *= -1
    # preds[preds > 1] = 1 - (preds[preds > 1] - 1)

    print(f"{name} Results:")
    # print("Predictions:", preds.numpy().flatten())
    # print("Labels:", labels.numpy().flatten())
    # print()
    print("MSE:", mean_squared_error(preds, labels))
    print("MAE:", mean_absolute_error(preds, labels))
    pearson = PearsonCorrCoef()
    print("R2:", pearson(preds, labels))
    results = np.hstack((preds, labels))
    np. savetxt(name+"-result.txt", results)


if __name__ == "__main__":
    # Load data
    train_loader, val_loader= load_data(computation=False)

    # Create model, optimizer, and criterion
    model = create_model()
    optimizer = create_optimizer(model)
    criterion = create_criterion()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_x1, batch_x2, batch_x3, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x1, batch_x2, batch_x3)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Get predictions and labels
    train_preds, train_labels = evaluate(model, train_loader)
    val_preds, val_labels = evaluate(model, val_loader)
    #test_preds, test_labels = evaluate(model, test_loader)

    # save model
    save_model(model, path="model.pth")

    print_results(train_preds, train_labels, "Training")
    print_results(val_preds, val_labels, "Validation")
    #print_results(test_preds, test_labels, "Test")
