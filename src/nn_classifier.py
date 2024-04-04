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
    Artificial Neural Network (ANN) model for classification.

    Args:
        input (int): Number of input features.
        hidden (int): Number of hidden units.
        output (int): Number of output classes.

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
            nn.ReLU(),
            nn.Linear(hidden, output),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Forward pass through the network
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, verbose=1):
    """
    Trains a neural network model using the given dataloader, loss function, and optimizer.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        verbose (int, optional): Verbosity level. Set to 1 to print basic progress updates, 2 to print
            additional loss updates every 5 batches, and 3 to print loss updates for every batch. Defaults to 1.

    Returns:
        list: The learning curve, which is a list of loss values for each batch during training.
        model: The trained model.
    """
    size = len(dataloader.dataset)
    model.train()

    learning_curve = []

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)  # .squeeze() # Reshape prediction to match the shape of y
        y = y.view_as(pred)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        learning_curve.append(loss.item())

        if verbose >= 3:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Batch {batch}: Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if verbose == 2 and batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    if verbose >= 1:  # Basic progress update after all batches
        print(f"Training complete. Final loss: {learning_curve[-1]:>7f}")

    return learning_curve, model


def test(dataloader, model, loss_fn, verbose):
    num_batches = len(dataloader)
    model.eval()
    test_error = 0
    test_loss, current = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = y.view_as(pred)
            test_loss = loss_fn(pred, y).item()
            test_error += test_loss
            current += len(X)
            if verbose >= 3:  # Detailed output for each batch
                print(f"Batch Test Loss: {test_loss:>8f}")

    test_error /= num_batches
    if verbose >= 1:  # Basic summary of test results
        print(f"Average Test MSE: {test_error:>8f} \n")

    return test_error


def train_and_eval(
    model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=5, verbose=1
):
    """
    Train and evaluate a neural network model.

    Args:
        model (torch.nn.Module): The neural network model to train and evaluate.
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The data loader for the testing dataset.
        loss_fn: The loss function to use for training and evaluation.
        optimizer: The optimizer to use for training.
        epochs (int): The number of epochs to train the model (default: 5).
        verbose (int): The verbosity level (0 - no output, 1 - print epoch information, 2 - print detailed information for each iteration) (default: 1).

    Returns:
        tuple: A tuple containing the best error rate, the best model, and the learning curve.

    """
    best_test_err = float("inf")
    best_model = None
    learning_curve_epochs = []

    for t in range(epochs):
        if verbose >= 1:
            print(f"Epoch {t+1}\n-------------------------------")
        learning_curve, _ = train(
            train_dataloader, model, loss_fn, optimizer, verbose=verbose
        )
        learning_curve_epochs.extend(learning_curve)

        # add curve from individual epoch to learning_curve_epochs

        # Test the model after all epochs
        err_rate = test(test_dataloader, model, loss_fn, verbose=verbose)

        # Save the best model
        if err_rate < best_test_err:
            best_err_rate = err_rate
            best_model = model

    if verbose >= 1:  # Final summary after all epochs
        print(
            f"Finished Training training the neuron. Best Test loss: {best_test_err:>8f}\n"
        )
        print("___" * 20, "\n")

    return best_err_rate, best_model, learning_curve_epochs


def nested_layer(
    dataset,
    hidden_neurons,
    batch_size,
    standardize=False,
    epochs=5,
    verbose=1,
    K_inner=5,
):
    # Convert dataset to tensors
    X = dataset[0]
    y = dataset[1]

    # Initialize variables to store results
    inner_k_folds = model_selection.KFold(n_splits=K_inner, shuffle=True)

    # Create np.arrays to store results
    best_evaluation_error = np.empty((K_inner, 1))
    optimal_hidden_neurons = np.empty((K_inner, 1))
    optimal_model_per_fold = np.empty((K_inner, 1), dtype=object)
    learning_curves_per_fold = []

    # Create np.array with None values to store lowest error for each hidden neuron size
    hidden_neurons_best_error = np.empty((len(hidden_neurons), 1))
    hidden_neurons_best_error.fill(float("inf"))

    inner_fold_best_model = None
    inner_fold_optimal_hidden_neurons = None
    inner_fold_lowest_error = float("inf")

    # Inner k-fold loop
    for i, (train_indx, test_index) in enumerate(inner_k_folds.split(X, y)):
        X_train, y_train = X[train_indx, :], y[train_indx]
        X_test, y_test = X[test_index, :], y[test_index]

        # Standardize the data
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Create dataset objects
        train_dataset = NNDataset(X_train, y_train)
        test_dataset = NNDataset(X_test, y_test)

        # Create dataloaders for training and testing
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        if verbose >= 2:
            print(f"Starting Inner Fold {i+1}/{K_inner}", "\n", "-" * 30)

        fold_best_error = float("inf")
        fold_optimal_hidden_neurons = None
        fold_best_model = None
        fold_best_learning_curve = None

        sample_features, _ = train_dataset[0]  # Get a sample to determine feature size
        input_size = sample_features.shape[0]  # Assuming features are a flat tensor

        # Evaluate different num of hidden neurons
        for j, num_neurons in enumerate(hidden_neurons):
            if verbose >= 1:
                print(f"Evaluating {num_neurons} neurons in Inner Fold {i+1}/{K_inner}")

            # Create model, loss function and optimizer for each configuration of hidden neurons
            model = ANN(input=input_size, hidden=num_neurons, output=1)
            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Return best model, best test error and learning curve
            val_err_rate, model, learning_curve = train_and_eval(
                model,
                train_dataloader,
                test_dataloader,
                loss_fn,
                optimizer,
                epochs=epochs,
                verbose=verbose,
            )

            # Save the best model and error rate for the current fold
            if val_err_rate < fold_best_error:
                fold_best_error = val_err_rate
                fold_optimal_hidden_neurons = num_neurons
                fold_best_model = model
                fold_best_learning_curve = learning_curve

            # Save the best training error for the current number of neurons
            if val_err_rate < hidden_neurons_best_error[j]:
                hidden_neurons_best_error[j] = val_err_rate

        # Save information from each fold
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


def error_rate(X, y, model, verbose):
    """
    Calculate the classification error rate for a given model.

    Parameters:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Target labels.
    - model (torch.nn.Module): Trained model.
    - verbose (int): Verbosity level. If greater than or equal to 1, print accuracy and error rate.

    Returns:
    - accuracy (float): Classification accuracy.
    - error_rate (float): Classification error rate.
    """
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize list to store all predictions
    all_preds = []

    # Set model to evaluation mode
    model.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Make predictions
        pred = model(X)
        pred = pred.round().squeeze()  # Assuming binary classification

        # Store all predictions (used for statistical analysis)
        all_preds.extend(pred.tolist())

        # Assuming binary classification
        accuracy = pred.eq(y).sum().item() / len(y)
        error_rate = 1 - accuracy

    if verbose >= 1:
        print(f"Accuracy: {accuracy:.8f}")
        print(f"Error rate: {error_rate:.8f}")

    return accuracy, error_rate, np.asanyarray(all_preds)


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
    learning_curve_list = []

    # Convert training and test sets to tensors
    X_train_tensor = torch.tensor(train_set[0], dtype=torch.float32)
    y_train_tensor = torch.tensor(train_set[1], dtype=torch.float32)

    # Create dataset objects
    train_dataset = NNDataset(X_train_tensor, y_train_tensor)

    # Create dataloaders for the training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get a sample to determine feature size
    sample_features, _ = train_dataset[0]
    input_size = sample_features.shape[0]

    # Initialize the neural network model, loss function, and optimizer
    model = ANN(input=input_size, hidden=hidden_neuron, output=1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        learning_curve, model = train(
            train_dataloader, model, loss_fn, optimizer, verbose=verbose
        )
        learning_curve_list.extend(learning_curve)

    _, test_err, squared_err = error_rate(test_set[0], test_set[1], model, verbose)
    # Evaluate the model on the test set
    # TODO
    # Return the error_rate function (squared error and error)
    return test_err, learning_curve_list, squared_err
