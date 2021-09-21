import torch.nn as nn
import numpy as np

from predict import predict_validation


def train(model, train_dataset, val_dataset, criterion, optimizer, batch_size, epochs=10, save_model_filepath=None):
    model.train()

    iteration = 0
    min_val_loss = float('inf')
    for e in range(epochs):
        hidden = model.init_hidden(batch_size)

        train_losses = []
        for i, (inputs, targets) in enumerate(train_dataset):
            iteration += 1
            optimizer.zero_grad()

            outputs, hidden = model(inputs, (hidden[0].detach(), hidden[1].detach()))

            loss = criterion(outputs, targets.float())
            train_losses.append(loss.item())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # Obtain validation results
            if iteration % 50 == 0:
                val_losses = predict_validation(model, val_dataset, criterion, batch_size)

                train_loss_mean = np.mean(train_losses)
                val_loss_mean = np.mean(val_losses)

                # Save model with the lowest validation loss
                if (min_val_loss > val_loss_mean):
                    min_val_loss = val_loss_mean

                    if save_model_filepath is not None:
                        model.save_model(save_model_filepath)

                print(
                    "Epoch: {},".format(e),
                    "Iteration: {},".format(iteration),
                    "Train Loss: {},".format(train_loss_mean),
                    "Val Loss: {}".format(val_loss_mean),
                    "Min. Val Loss: {}".format(min_val_loss)
                )