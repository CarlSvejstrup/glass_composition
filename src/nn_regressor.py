import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# ANN
import torch
from torch import nn, utils
from torch.utils.data import DataLoader, Dataset, Subset


from sklearn import model_selection
import sklearn.linear_model as lm
from dtuimldmtools import rlr_validate


class NNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class ANN(nn.Module):
    """
    Artificial Neural Network (ANN) model for regression tasks.

    Args:
        input (int): Number of input features.
        hidden (int): Number of hidden units in the network.
        output (int): Number of output units.

    Attributes:
        flatten (nn.Flatten): Flattens the input tensor.
        linear_relu_stack (nn.Sequential): Sequential container for the linear and activation layers.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self, input, hidden, output):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ELU(),
            nn.Linear(hidden, output),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, verbose=1):
    """
    Trains the neural network model using the given dataloader, loss function, and optimizer.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        verbose (int, optional): Verbosity level. Set to 1 to print basic progress update, 2 to print loss after every 5 batches, and 3 to print loss after every batch. Defaults to 1.

    Returns:
        list: The learning curve, which contains the loss values for each batch during training.
    """
    # Get the size of the dataset
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()

    # Initialize an empty list to store the learning curve
    learning_curve = []

    # Iterate over each batch in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Perform forward pass through the model
        pred = model(X)
        # Reshape the target tensor to match the shape of the prediction tensor
        y = y.view_as(pred)
        # Compute the loss
        loss = loss_fn(pred, y)

        # Perform backpropagation and update the model's parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Append the current loss value to the learning curve list
        learning_curve.append(loss.item())

        # Print detailed loss information if verbose level is 3
        if verbose >= 3:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Batch {batch}: Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Print loss information every 5 batches if verbose level is 2
        if verbose == 2 and batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Print basic progress update after all batches if verbose level is 1
    if verbose >= 1:
        print(f"Training complete. Final loss: {learning_curve[-1]:>7f}")

    # Return the learning curve
    return [np.mean(learning_curve)]


def test(dataloader, model, loss_fn, verbose=1):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): The test dataloader.
        model (torch.nn.Module): The trained model.
        loss_fn (torch.nn.Module): The loss function used for evaluation.
        verbose (int, optional): Verbosity level. Default is 1.

    Returns:
        tuple: A tuple containing the average test error (MSE) and an array of squared errors.

    """
    # Get the size of the test dataset
    size = len(dataloader.dataset)
    # Get the number of batches in the dataloader
    num_batches = len(dataloader)
    # Set the model to evaluation mode
    model.eval()
    # Initialize test error and loss variables
    test_error = 0
    test_loss, current = 0, 0

    # Initialize an empty list to store squared errors
    all_squared_errors = []
    all_predictions = []
    # Disable gradient calculation during evaluation
    with torch.no_grad():
        # Iterate over the test dataloader
        for X, y in dataloader:
            # Forward pass through the model
            pred = model(X)
            all_predictions.extend(pred.view(-1).tolist())

            # Reshape the target tensor to match the predicted tensor shape
            y = y.view_as(pred)
            # Calculate the loss between the predicted and target tensors
            test_loss = loss_fn(pred, y).item()
            # Update the test error
            test_error += test_loss

            # Calculate the squared errors
            squared_errors = np.square(y - pred)
            # Flatten the squared errors tensor and convert to a list, then extend the all_squared_errors list
            all_squared_errors.extend(squared_errors.view(-1).tolist())

            # Update the current count of processed samples
            current += len(X)
            # Print detailed output for each batch if verbosity level is 3 or higher
            if verbose >= 3:
                print(f"Batch Test Loss: {test_loss:>8f}")

    # Calculate the average test error
    test_error /= num_batches
    # Print basic summary of test results if verbosity level is 1 or higher
    if verbose >= 1:
        print(f"Average Test MSE: {test_error:>8f} \n")

    # Return the average test error and the array of squared errors
    return test_error, np.asarray(all_squared_errors), np.asarray(all_predictions)


def train_and_eval(
    model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=5, verbose=1
):
    """
    Trains and evaluates a neural network model.

    Args:
        model (torch.nn.Module): The neural network model to train and evaluate.
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss_fn (torch.nn.Module): The loss function to use for training and evaluation.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        epochs (int, optional): The number of epochs to train the model (default: 5).
        verbose (int, optional): The verbosity level (0 - no output, 1 - print epoch information, default: 1).

    Returns:
        tuple: A tuple containing the best test error, the best model, and the learning curve.

    """
    best_test_error = float("inf")
    best_model = None
    learning_curve_epochs = []

    # See dtype of dataloader

    for t in range(epochs):
        if verbose >= 1:
            print(f"Epoch {t+1}\n-------------------------------")
        learning_curve = train(
            train_dataloader, model, loss_fn, optimizer, verbose=verbose
        )
        learning_curve_epochs.extend(learning_curve)

    
        # Test the model after each epoch
        eval_error, squared_err, all_predictions = test(
            test_dataloader, model, loss_fn, verbose=verbose
        )

        # Save the best model
        if eval_error < best_test_error:
            best_test_error = eval_error
            best_model = model

    if verbose >= 1:  # Final summary after all epochs
        print(f"Finished Training the model. Best Test MSE: {best_test_error:>8f}\n")
        print("___" * 20, "\n")

    return (
        best_test_error,
        best_model,
        learning_curve_epochs,
        squared_err,
        all_predictions,
    )


def nested_layer(
    dataset,
    hidden_neurons,
    batch_size,
    epochs=5,
    verbose=1,
    K_inner=5,
):
    """
    Perform nested layer cross-validation for training and evaluating a neural network model.

    Args:
        dataset (tuple): A tuple containing the input features and target values.
        hidden_neurons (list): A list of integers representing the number of neurons in each hidden layer.
        batch_size (int): The batch size for training the neural network.
        epochs (int, optional): The number of epochs for training the neural network. Defaults to 5.
        verbose (int, optional): The verbosity level. Set to 0 for no output, 1 for minimal output, and 2 for detailed output. Defaults to 1.
        K_inner (int, optional): The number of inner folds for cross-validation. Defaults to 5.

    Returns:
        tuple: A tuple containing the optimal number of hidden neurons, the best model, the lowest error, and the best training error for each number of hidden neurons.
    """
    # Convert dataset to tensors
    X = dataset[0]
    y = dataset[1]

    # Initialize inner k-fold cross-validation
    inner_k_folds = model_selection.KFold(n_splits=K_inner, shuffle=True)

    # Arrays for storing results for neural network evaluation
    best_evaluation_error = np.empty((K_inner, 1))
    optimal_hidden_neurons = np.empty((K_inner, 1))
    optimal_model_per_fold = np.empty((K_inner, 1), dtype=object)
    learning_curves_per_fold = []
    hidden_neurons_best_error = np.empty((len(hidden_neurons), 1))
    hidden_neurons_best_error.fill(float("inf"))

    # Initialize variables to store the best model and lowest error across inner folds
    inner_fold_best_model = None
    inner_fold_optimal_hidden_neurons = None
    inner_fold_lowest_error = float("inf")

    # Inner k-fold loop
    for i, (train_indx, test_index) in enumerate(inner_k_folds.split(X, y)):
        X_train, y_train = X[train_indx, :], y[train_indx]
        X_test, y_test = X[test_index, :], y[test_index]

        # Standardize the data and prepare the dataloaders

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = NNDataset(X_train, y_train)
        test_dataset = NNDataset(X_test, y_test)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Print information about the current inner fold if verbosity level is 2 or higher
        if verbose >= 2:
            print(f"Starting Inner Fold {i+1}/{K_inner}", "\n", "-" * 30)

        # Get a sample to determine feature size
        sample_features, _ = train_dataset[0]
        input_size = sample_features.shape[0]

        # Initialize variables to store the best error and model for the current fold
        fold_best_error = float("inf")
        fold_optimal_hidden_neurons = None
        fold_best_model = None
        fold_best_learning_curve = None

        # Evaluate different num of hidden neurons
        for j, num_neurons in enumerate(hidden_neurons):
            if verbose >= 1:
                print(f"Evaluating {num_neurons} neurons in Inner Fold {i+1}/{K_inner}")

            # Initialize the neural network model, loss function, and optimizer for each hidden neuron configuration
            model = ANN(input=input_size, hidden=num_neurons, output=1)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Return best model, best test error and learning curve
            test_error, model, learning_curve, _, _ = train_and_eval(
                model,
                train_dataloader,
                test_dataloader,
                loss_fn,
                optimizer,
                epochs=epochs,
                verbose=verbose,
            )

            # Save the best model and error for the current number of neurons
            if test_error < fold_best_error:
                fold_best_error = test_error
                fold_optimal_hidden_neurons = num_neurons
                fold_best_model = model
                fold_best_learning_curve = learning_curve

            # Save the best training error for the current number of neurons
            if test_error < hidden_neurons_best_error[j]:
                hidden_neurons_best_error[j] = test_error

        # Save information from each inner fold
        best_evaluation_error[i] = fold_best_error
        optimal_hidden_neurons[i] = fold_optimal_hidden_neurons
        optimal_model_per_fold[i] = fold_best_model
        learning_curves_per_fold.append(fold_best_learning_curve)

        if verbose >= 1:
            print("###" * 20)
            print(
                f"Inner Fold {i+1}/{K_inner}: Best Hidden Neurons: {fold_optimal_hidden_neurons}, Test Error: {fold_best_error}",
                "\n",
                "###" * 20,
                "\n",
            )

        # Save the best model from the inner loop
        if fold_best_error < inner_fold_lowest_error:
            inner_fold_lowest_error = fold_best_error
            inner_fold_best_model = fold_best_model
            inner_fold_optimal_hidden_neurons = fold_optimal_hidden_neurons

    if verbose >= 1:
        print("**" * 20)
        print(
            f"\nBest Model across Inner Folds: {inner_fold_optimal_hidden_neurons} Neurons, Lowest Error: {inner_fold_lowest_error}"
        )
        print("**" * 20, "\n")

    return (
        inner_fold_optimal_hidden_neurons,
        inner_fold_best_model,
        inner_fold_lowest_error,
        hidden_neurons_best_error,
    )


def outer_test(train_set, test_set, hidden_neuron, batch_size, epochs=5, verbose=1):
    """
    Perform the outer loop of the nested cross-validation for training and evaluating a neural network model.

    Args:
        train_set (tuple): A tuple containing the input features and target values for training.
        test_set (tuple): A tuple containing the input features and target values for testing.
        hidden_neuron (int): The number of hidden neurons in the neural network.
        batch_size (int): The batch size for training the neural network.
        epochs (int, optional): The number of epochs for training the neural network. Defaults to 5.
        verbose (int, optional): The verbosity level. Set to 0 for no output, 1 for minimal output, and 2 for detailed output. Defaults to 1.

    Returns:
        tuple: A tuple containing the best test error, the best model, and the learning curve.

    """
    if verbose >= 1:
        print("Starting Outer Loop training", "\n", "-" * 30)

    # Convert training and test sets to tensors
    X_train_tensor = torch.tensor(train_set[0], dtype=torch.float32)
    y_train_tensor = torch.tensor(train_set[1], dtype=torch.float32)
    X_test_tensor = torch.tensor(test_set[0], dtype=torch.float32)
    y_test_tensor = torch.tensor(test_set[1], dtype=torch.float32)

    # Create dataset objects
    train_dataset = NNDataset(X_train_tensor, y_train_tensor)
    test_dataset = NNDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders for the training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Get a sample to determine feature size
    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[0]

    # Initialize the neural network model, loss function, and optimizer
    model = ANN(input=input_size, hidden=hidden_neuron, output=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train and evaluate the model
    test_error, model, learning_curve, squared_err, all_predictions = train_and_eval(
        model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        epochs=epochs,
        verbose=verbose,
    )

    return test_error, model, learning_curve, squared_err, all_predictions
