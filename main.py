import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import time
import math

from dataset import Dataset
from model import Model
from train import train
from predict import predict_test

def build_train_validation_split_batches(dataset, batch_size):
    padded_input_sequences = dataset.pad_input_sequences()
    targets = dataset.targets

    dataset = TensorDataset(torch.from_numpy(np.array(padded_input_sequences)), torch.from_numpy(np.array(targets)))

    # Get training, validation datasets
    train_size = math.floor(len(dataset)*0.8)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # create data loaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, valid_loader


def build_train_validation_test_split_batches(dataset, batch_size):
    # Input and target data
    padded_input_sequences = dataset.pad_input_sequences()
    targets = dataset.targets

    # Get training, validation, and testing datasets
    train_dataset, val_dataset, test_dataset = dataset.get_train_val_test_split(padded_input_sequences, targets, 0.2, 0.5)

    # Create Tensor Datasets
    train_data = TensorDataset(torch.from_numpy(np.array(train_dataset[0])), torch.from_numpy(np.array(train_dataset[1])))
    val_data = TensorDataset(torch.from_numpy(np.array(val_dataset[0])), torch.from_numpy(np.array(val_dataset[1])))
    test_data = TensorDataset(torch.from_numpy(np.array(test_dataset[0])), torch.from_numpy(np.array(test_dataset[1])))

    # create data loaders
    # Note: drop_last -> drops last incomplete batch
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, valid_loader, test_loader

def train_model(model, criterion, optimizer, batch_size, train_loader, valid_loader, save_filepath=None,
                print_time_elapsed=False):

    start = time.time()
    train(model, train_loader, valid_loader, criterion, optimizer, batch_size, save_model_filepath=save_filepath)
    end = time.time()

    if print_time_elapsed:
        print("Training Time Elapsed: {} seconds".format(round(end-start, 2)))


def test_model(dataset, embedding_dim, num_layers, hidden_size, output_size, save_filepath, test_loader,
                     criterion, batch_size, print_time_elapsed=False):
    # Load Saved Model
    model_loaded = Model(dataset, embedding_dim, num_layers, hidden_size, output_size)
    model_loaded.load_state_dict(torch.load(save_filepath))

    # Apply model against the test dataset,
    start = time.time()
    predict_test(model_loaded, test_loader, criterion, batch_size)
    end = time.time()

    if print_time_elapsed:
        print("Testing Time Elapsed: {} seconds".format(round(end-start, 2)))

    ### Predicting Labels for New Text (Not from Dataset)
    print()

    texts = ["Is my computer going to be okay?", "Does this work for Mac?", "How many rams do I need?",
             "Is this compatible with windows?", "What is the minimum requirements?", "Why should I buy this?",
             "Is shipping free for this item?",
             "It's been 6 days and I have not received my order. Can I get a refund?"]

    for text in texts:
        print(text)
        print(model_loaded.make_text_prediction(text))
        print()



def train_validate_test():
    # Dataset
    dataset = Dataset()
    batch_size = 50

    # Model
    embedding_dim = 128
    num_layers = 1
    hidden_size = 256
    output_size = 1

    save_dir = './models/{}'
    model_name = "model"
    save_filename = "{}_{}_{}_{}.pt".format(model_name, embedding_dim, num_layers, hidden_size)
    save_filepath = save_dir.format(save_filename)

    train_loader, valid_loader, test_loader = build_train_validation_test_split_batches(dataset, batch_size)

    model = Model(dataset, embedding_dim, num_layers, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    train_model(model, criterion, optimizer, batch_size, train_loader, valid_loader, save_filepath=save_filepath,
                print_time_elapsed=True)
    print()

    # Test Model
    test_model(dataset, embedding_dim, num_layers, hidden_size, output_size, save_filepath, test_loader,
               criterion, batch_size, print_time_elapsed=True)


def train_validate():
    # Dataset
    dataset = Dataset()
    batch_size = 50

    # Model
    embedding_dim = 128
    num_layers = 1
    hidden_size = 256
    output_size = 1

    save_dir = './models/{}'
    model_name = "model"
    save_filename = "{}_{}_{}_{}_test.pt".format(model_name, embedding_dim, num_layers, hidden_size)
    save_filepath = save_dir.format(save_filename)

    train_loader, valid_loader = build_train_validation_split_batches(dataset, batch_size)

    model = Model(dataset, embedding_dim, num_layers, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    train_model(model, criterion, optimizer, batch_size, train_loader, valid_loader, save_filepath=save_filepath,
                print_time_elapsed=True)


if __name__ == "__main__":
    train_validate()