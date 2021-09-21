import torch
import numpy as np


def predict_validation(model, val_dataset, criterion, batch_size):
    model.eval()
    hidden = model.init_hidden(batch_size)

    losses = []
    for i, (inputs, targets) in enumerate(val_dataset):
        outputs, hidden = model(inputs, (hidden[0].detach(), hidden[1].detach()))
        loss = criterion(outputs, targets.float())
        losses.append(loss.item())
    return losses


def predict_test(model, test_dataset, criterion, batch_size):
    model.eval()
    hidden = model.init_hidden(batch_size)

    losses = []
    num_corrects = 0
    for i, (inputs, targets) in enumerate(test_dataset):
        outputs, hidden = model(inputs, (hidden[0].detach(), hidden[1].detach()))
        loss = criterion(outputs, targets.float())
        losses.append(loss.item())

        # Obtain classification accuracy results
        pred = torch.round(outputs.squeeze())
        correct_tensor = pred.eq(targets.float().view_as(pred))  #
        correct = np.squeeze(correct_tensor.numpy())
        num_corrects += np.sum(correct)

    print("Test Classification Accuracy: {} %".format((num_corrects / len(test_dataset.dataset)) * 100))
    print("Average Loss: {}".format(np.mean(losses)))