import os  
import time
import numpy as np  
import pandas as pd  

import torch  
import torch.nn as nn  
import torch.optim as optim 
import torch.nn.functional as F

from absl import logging

from scipy.stats import shapiro 

from skorch import NeuralNetRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import RandomizedSearchCV  


class L2PModel(nn.Module):  
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_rate, 
        num_layers=12
    ):  
        """  
        Initialize the enhanced L2PModel model with increased complexity.  

        Parameters:  
            input_size (int): Number of input features.  
            hidden_size (int): Number of neurons in each hidden layer.  
            output_size (int): Number of outputs (e.g., 1 for one regression task).  
            dropout_rate (float): Dropout rate for regularization.  
            num_layers (int): Number of hidden layers to include (>=3 for increased depth).  
        """  
        super(L2PModel, self).__init__()  
        
        # Input Layer  
        self.input_layer = nn.Linear(input_size, hidden_size)  
        self.bn_input = nn.BatchNorm1d(hidden_size)  # Batch Normalization for input layer  

        # Hidden Layers  
        self.hidden_layers = nn.ModuleList()  
        for _ in range(num_layers - 1):  # Add n-1 hidden layers  
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))  
        
        self.bn_hidden = nn.ModuleList()  
        for _ in range(num_layers - 1):  # Add batch norm for each hidden layer  
            self.bn_hidden.append(nn.BatchNorm1d(hidden_size))  
        
        # Output Layer  
        self.output_layer = nn.Linear(hidden_size, output_size)  

        # Dropout  
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(
        self, 
        x
    ):  
        """  
        Define the forward pass of the L2PModel.  

        Parameters:  
            x (torch.Tensor): Input tensor.  

        Returns:  
            torch.Tensor: Output tensor of the L2PModel.  
        """  
        # Input Layer  
        x = F.relu(self.bn_input(self.input_layer(x)))  
        x = self.dropout(x)  

        # Hidden Layers with residual connections  
        for i, hidden_layer in enumerate(self.hidden_layers):  
            residual = x  # Store input of the layer  
            x = F.relu(self.bn_hidden[i](hidden_layer(x)))  
            x = self.dropout(x)  

            # Add residual connection  
            x += residual  
        
        # Output Layer  
        return self.output_layer(x)  
 

def validate_model_architecture(
    model, 
    input_size, 
    hidden_size, 
    output_size, 
    dropout_rate, 
    num_layers
):  
    """  
    Validates that the model's architecture matches the provided parameters.  

    Parameters:  
        model (torch.nn.Module): The model to validate.  
        input_size (int): Expected input size.  
        hidden_size (int): Expected hidden layer size.  
        output_size (int): Expected output size.  
        dropout_rate (float): Expected dropout rate.  
        num_layers (int): Expected number of layers in the model.  

    Raises:  
        ValueError: If the model architecture does not match the provided parameters.  
    """  
    # Validate the input layer  
    if model.input_layer.in_features != input_size:  
        logging.error(f"Input layer in_features {model.input_layer.in_features} does not match input_size {input_size}.") 
        exit(1)
    if model.input_layer.out_features != hidden_size:  
        logging.error(f"Input layer out_features {model.input_layer.out_features} does not match hidden_size {hidden_size}.")  
        exit(1)

    # Validate BatchNorm for input layer  
    if model.bn_input.num_features != hidden_size:  
        logging.error(f"BatchNorm num_features for input layer {model.bn_input.num_features} does not match hidden_size {hidden_size}.")  
        exit(1)

    # Validate hidden layers  
    if len(model.hidden_layers) != num_layers - 1:  
        logging.error(f"Model has {len(model.hidden_layers)} hidden layers, but {num_layers - 1} were expected.")  
        exit(1)

    for i, hidden_layer in enumerate(model.hidden_layers):  
        if hidden_layer.in_features != hidden_size or hidden_layer.out_features != hidden_size:  
            logging.error(f"Hidden Layer {i + 1} size mismatch: "  
                             f"in_features={hidden_layer.in_features}, out_features={hidden_layer.out_features}, "  
                             f"expected={hidden_size} for both.")  
            exit(1)
        # Validate batch normalization for each hidden layer  
        if model.bn_hidden[i].num_features != hidden_size:  
            logging.error(f"BatchNorm num_features for Hidden Layer {i + 1} does not match hidden_size {hidden_size}.")  
            exit(1)

    # Validate dropout rate  
    if model.dropout.p != dropout_rate:  
        logging.error(f"Dropout rate {model.dropout.p} does not match expected dropout_rate {dropout_rate}.")  
        exit(1)

    # Validate the output layer  
    if model.output_layer.in_features != hidden_size:  
        logging.error(f"Output layer in_features {model.output_layer.in_features} does not match hidden_size {hidden_size}.")  
        exit(1)
    if model.output_layer.out_features != output_size:  
        logging.error(f"Output layer out_features {model.output_layer.out_features} does not match output_size {output_size}.")  
        exit(1)


def freeze_layers(
    model, 
    freeze_up_to=6
):  
    """  
    Freezes the layers of the model up to a specific index (exclusive).  

    Parameters:  
        model (L2PModel): The model whose layers will be frozen.  
        freeze_up_to (int): The number of layers to freeze (including the input layer   
                            and batch normalization).  
    """  
    # Freeze input layer and its batch norm  
    for param in model.input_layer.parameters():  
        param.requires_grad = False  
    for param in model.bn_input.parameters():  
        param.requires_grad = False  

    # Freeze the first `freeze_up_to` hidden layers  
    for i in range(freeze_up_to - 1):  # -1 because input_layer is already frozen  
        for param in model.hidden_layers[i].parameters():  
            param.requires_grad = False  
        for param in model.bn_hidden[i].parameters():  
            param.requires_grad = False  

    logging.info(f"Froze the input layer and first {freeze_up_to} hidden layers.")  


def search_hyperparameters(  
    csv_file,  
    exclude_columns=None,  
    target_column=None, 
    output_size=1,   
    model=None,  
    param_dist=None,  
    search_method='randomized',  # 'randomized' or 'grid'  
    n_iter=10,  # Only used for RandomizedSearchCV  
    cv=3,  
    random_state=42  
):  
    """  
    Perform hyperparameter tuning on a regression model using RandomizedSearchCV or GridSearchCV and skorch.  

    Parameters:  
        csv_file (str): Path to the CSV file containing the dataset.  
        exclude_columns (list): List of columns to exclude from the features. Default is None.  
        target_column (str): Name of the target column. Default is None.  
        output_size (int): Expected output size.  
        model (nn.Module): PyTorch model to use for training. Default is None.  
        param_dist (dict): Dictionary of hyperparameters to search. Default is None.  
        search_method (str): Search method to use ('randomized' or 'grid'). Default is 'randomized'.  
        n_iter (int): Number of random combinations to try (only for RandomizedSearchCV). Default is 10.  
        cv (int): Number of cross-validation folds. Default is 3.  
        random_state (int): Random seed for reproducibility. Default is 42.  

    Returns:  
        dict: Best hyperparameters found.  
        float: Best validation score (negative MSE).  

    Raises:  
        ValueError: If any validation check fails.  
    """  
    # Validate CSV file  
    if not os.path.exists(csv_file):  
        logging.error(f"CSV file '{csv_file}' does not exist.")  
    if not csv_file.endswith('.csv'):  
        logging.error(f"File '{csv_file}' is not a CSV file.")  

    # Load the dataset from the CSV file  
    try:  
        data = pd.read_csv(csv_file)  
    except Exception as e:  
        logging.error(f"Failed to load CSV file '{csv_file}': {e}")  

    # Validate target column  
    if target_column is None:  
        logging.error("Target column must be specified.")  
    if target_column not in data.columns:  
        logging.error(f"Target column '{target_column}' does not exist in the dataset.")  

    # Validate exclude columns  
    if exclude_columns is not None:  
        for col in exclude_columns:  
            if col not in data.columns:  
                logging.error(f"Exclude column '{col}' does not exist in the dataset.")  
        if target_column in exclude_columns:  
            logging.error("Target column cannot be in exclude_columns.")  

    # Validate model  
    if model is None:  
        logging.error("Model must be provided.")  

    # Validate hyperparameter search space  
    if param_dist is None:  
        logging.error("Hyperparameter search space (param_dist) must be provided.")  
    if not isinstance(param_dist, dict):  
        logging.error("Hyperparameter search space (param_dist) must be a dictionary.")  

    # Exclude specified columns and separate features (X) and target (y)  
    if exclude_columns:  
        X = data.drop(columns=exclude_columns + [target_column])  
    else:  
        X = data.drop(columns=[target_column])  
    y = data[target_column]  

    # Ensure inputs are in the correct format  
    X = X.astype(np.float32).values  
    y = y.astype(np.float32).values.reshape(-1, 1)  

    # Wrap the PyTorch model in a skorch regressor  
    net = NeuralNetRegressor(  
            module=model,  
            module__input_size=X.shape[1],  # Input size is determined by the number of features  
            module__output_size=output_size,  # Regression output  
            criterion=nn.MSELoss,  
            optimizer=optim.Adam,  
            max_epochs=20,  # Default number of epochs  
            batch_size=32,  # Default batch size  
            verbose=0,  # Suppress output  
    )  

    # Perform hyperparameter search  
    if search_method == 'randomized':  
        search = RandomizedSearchCV(  
                    estimator=net,  
                    param_distributions=param_dist,  # Hyperparameter search space  
                    n_iter=n_iter,  # Number of random combinations to try  
                    cv=cv,  # Number of cross-validation folds  
                    scoring='neg_mean_squared_error',  # Use negative MSE as the evaluation metric  
                    verbose=1,  # Print progress  
                    random_state=random_state,  # Random seed for reproducibility  
        )  
    elif search_method == 'grid':  
        search = GridSearchCV(  
                    estimator=net,  
                    param_grid=param_dist,  # Hyperparameter search space  
                    cv=cv,  # Number of cross-validation folds  
                    scoring='neg_mean_squared_error',  # Use negative MSE as the evaluation metric  
                    verbose=1,  # Print progress  
        )  
    else:  
        logging.error("Invalid search_method. Choose 'randomized' or 'grid'.")  

    # Fit the search  
    search.fit(X, y)  

    # Return the best hyperparameters and corresponding score  
    return search.best_params_, search.best_score_  


def train_in_batches(
    model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    optimizer, 
    criterion,  
    num_epochs=1000, 
    use_patience=True, 
    patience=50, 
    model_dir="models",  
    model_name="best_model.pth", 
    save_model=False, 
    batch_size=32, 
    multi_objective=False
):  
    """  
    Train the model using the provided training and validation data in batches,  
    while utilizing GPU if available, supporting single or multi-objective regression.  

    Parameters:  
        model (torch.nn.Module): Model to be trained.  
        X_train (torch.Tensor): Training features.  
        y_train (torch.Tensor): Training labels (for single or multi-objective predictions).  
        X_val (torch.Tensor): Validation features.  
        y_val (torch.Tensor): Validation labels (for single or multi-objective predictions).  
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.  
        criterion (torch.nn.Module): Loss function.  
        num_epochs (int): Number of training epochs.    
        use_patience (bool): Apply early stopping if True.  
        patience (int): Patience for early stopping.  
        model_dir (str): Directory to save the best model.  
        model_name (str): Name of the saved model file.  
        save_model (bool): Save the best model if True.  
        batch_size (int): Size of each batch for training and validation.  
        multi_objective (bool): Whether the task is multi-objective regression.  

    Returns:  
        model (torch.nn.Module): Best trained model (with the lowest validation loss).  
        history (dict): Dictionary containing training and validation loss history.  
        wallclock (float): Elapsed wall clock time during training.  
        cputime (float): Elapsed CPU time during training.  
    """  

    # Ensure model directory exists, if saving  
    if save_model:  
        os.makedirs(model_dir, exist_ok=True)  

    # Set the device to GPU if available  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  

    # Move model and data to the device  
    model = model.to(device)  
    X_train, y_train = X_train.to(device), y_train.to(device)  
    X_val, y_val = X_val.to(device), y_val.to(device)  

    # Initialize variables  
    best_val_loss = float('inf')  # Best validation loss  
    patience_counter = 0  # Counter for early stopping  
    best_model_state = None  # To store the best model's weights  
    history = {"train_loss": [], "val_loss": []}  # History of losses  

    # Start instrumentation for timing  
    wallclock_start = time.perf_counter()  
    cputime_start = time.process_time()  

    for epoch in range(num_epochs):  
        model.train()  # Set the model in training mode  
        train_loss = 0.0  

        # Train in batches  
        for i in range(0, len(X_train), batch_size):  
            # Get batches for training  
            X_batch = X_train[i:i + batch_size]  
            y_batch = y_train[i:i + batch_size]  

            # Skip batch if batch size is too small for BatchNorm  
            if X_batch.size(0) < 2:  # Batch size too small for BatchNorm  
                logging.warning(f"Skipping batch with size {X_batch.size(0)} (too small for BatchNorm)")  
                continue  

            # Forward pass, backpropagation, and optimization  
            optimizer.zero_grad()  
            outputs = model(X_batch)  

            # Ensure correct handling of multiple objectives  
            if multi_objective:  
                # Ensure outputs and labels are the same size  
                assert outputs.size() == y_batch.size(), "Model outputs and labels must have the same size for multi-objective regression."  
            else:  
                # For single-objective tasks, outputs and y_batch should both have shape [batch_size, 1]  
                assert outputs.size(1) == 1, "For single-objective regression, the output dimension must be 1."  

            loss = criterion(outputs, y_batch)  
            loss.backward()  
            optimizer.step()  

            train_loss += loss.item()  

        # Average training loss over the epoch  
        train_batches = max(1, len(X_train) // batch_size)  # Avoid division by zero  
        avg_train_loss = train_loss / train_batches  
        history["train_loss"].append(avg_train_loss)  

        # Validation in batches  
        model.eval()  # Set the model in evaluation mode  
        val_loss = 0.0  
        with torch.no_grad():  
            for i in range(0, len(X_val), batch_size):  
                # Get batches for validation  
                X_batch_val = X_val[i:i + batch_size]  
                y_batch_val = y_val[i:i + batch_size]  

                # Skip batch if batch size is too small for BatchNorm  
                if X_batch_val.size(0) < 2:  # Batch size too small for BatchNorm  
                    logging.warning(f"Skipping batch with size {X_batch_val.size(0)} (too small for BatchNorm)")  
                    continue  

                # Forward pass to calculate validation loss  
                val_outputs = model(X_batch_val)  

                # Ensure correct handling of multiple objectives  
                if multi_objective:  
                    assert val_outputs.size() == y_batch_val.size(), "Validation outputs and labels must match dimensions for multi-objective regression."  
                else:  
                    assert val_outputs.size(1) == 1, "Validation output dimension must be 1 for single-objective regression."  

                val_loss += criterion(val_outputs, y_batch_val).item()  

        # Average validation loss over the epoch  
        val_batches = max(1, len(X_val) // batch_size)  # Avoid division by zero  
        avg_val_loss = val_loss / val_batches  
        history["val_loss"].append(avg_val_loss)  

        # Save the best model if validation loss improves  
        if avg_val_loss < best_val_loss:  
            best_val_loss = avg_val_loss  
            patience_counter = 0  # Reset patience counter  
            best_model_state = model.state_dict()  # Save model state  
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Validation loss improved)")  
            if save_model:  
                torch.save(best_model_state, os.path.join(model_dir, model_name))  
                logging.info(f"Best model saved to {os.path.join(model_dir, model_name)}")  
        else:  
            patience_counter += 1  # Increment the patience counter  

        # Early stopping logic  
        if use_patience and patience_counter >= patience:  
            logging.info(f"Early stopping at epoch {epoch + 1}")  
            break  

        # Print losses every few epochs  
        if (epoch + 1) % 10 == 0:  
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")  

    # Load the best model state  
    if best_model_state is not None:  
        model.load_state_dict(best_model_state)  
        logging.info("Best model state loaded.")  

    # Stop instrumentation  
    wallclock = time.perf_counter() - wallclock_start  
    cputime = time.process_time() - cputime_start  

    return model, history, wallclock, cputime  


def train_and_predict(
    data, 
    target_column=None, 
    hidden_size=64, 
    output_size=1,
    learning_rate=0.0001, 
    dropout_rate=0.5, 
    num_layers=12, 
    batch_size=16, 
    use_patience=True,  
    patience=200, 
    num_epochs=1000, 
    test_size=0.2,  
    val_size=0.2, 
    random_state=42, 
    model_dir="models",  
    model_name="best_model.pth", 
    save_model=False
):  
    """  
    Train a robust neural network for regression and return predictions and error metrics.  
    If test_size is 0.0, only train the model and return training history.  

    Parameters:  
        data: Dataframe containing the dataset.
        target_column (str): Column to use as the target (y).  
        hidden_size (int): Number of neurons in the hidden layers.  
        output_size (int): Number of outputs (e.g., 1 for one regression task).
        learning_rate (float): Learning rate for the optimizer.  
        dropout_rate (float): Dropout rate for regularization.   
        num_layers (int): Number of hidden layers to include (>=3 for increased depth).  
        batch_size (int): Size of each batch for training and validation.  
        use_patience (bool): Whether to use early stopping.   
        patience (int): Patience for early stopping.   
        num_epochs (int): Maximum number of training epochs.  
        test_size (float): Proportion of the dataset to include in the test split.  
        val_size (float): Proportion of the training set for validation.  
        random_state (int): Random state for reproducibility.  
        model_dir (str): Directory to save the best model.  
        model_name (str): Name of the saved model file.  
        save_model (bool): Save the best model if True.  

    Returns:  
        predictions (np.array): Predicted values for the test set (None if test_size is 0.0).  
        history (dict): Dictionary containing training and validation loss history.  
        elapsed_time (dict): Dictionary capturing elapsed wallclock and CPU time during training/test.  
    """  
    logging.info(f"Training and predicting ...")  

    # Select the device (GPU if available)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  

    # Prepare features (X) and target(s) (y)  
    if isinstance(target_column, str):  # If it's a single target column  
        if target_column not in data.columns:  
            logging.error(f"Target column '{target_column}' not found in the dataset.")  

        # Single target: `y` will be a 1D vector  
        X = data.drop(columns=[target_column]).values  # Drop the single target column  
        y = data[target_column].values  

    elif isinstance(target_column, list):  # If there are multiple target columns  
        for col in target_column:  
            if col not in data.columns:  
                logging.error(f"Target column '{col}' not found in the dataset.")  

        # Multiple targets: `y` will be a 2D matrix  
        X = data.drop(columns=target_column).values  # Drop all specified target columns  
        y = data[target_column].values  

    else:  # Raise an error if target_column is invalid  
        logging.error("'target_column' must be a string (for one target) or a list (for multiple targets).")  

    # Convert data to PyTorch tensors and move to the selected device  
    X = torch.tensor(X, dtype=torch.float32).to(device)  

    # If there is one target, `y` must be reshaped to match model expectations  
    if len(y.shape) == 1:  # Single target: Make `y` a 2D tensor with shape (num_samples, 1)  
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)  
    else:  # Multi-target: `y` already has the correct shape (num_samples, num_targets)  
        y = torch.tensor(y, dtype=torch.float32).to(device)  

    # Split data into training, validation, and test sets  
    if test_size > 0.0:  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)  
    else:  
        # Use the entire dataset for training and validation  
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)  
        X_test, y_test = None, None  

    # Initialize the model, loss function, and optimizer  
    input_size = X_train.shape[1]  # Number of features  
    model = L2PModel(input_size, hidden_size, output_size, dropout_rate, num_layers).to(device)  # Initialize the model and move to the device  
    criterion = nn.MSELoss()  # Loss function  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer for training  

    # Train the model using the train in batches function 
    model, history, training_wallclock, training_cputime = train_in_batches(  
                                                                model,   
                                                                X_train,   
                                                                y_train,   
                                                                X_val,   
                                                                y_val,   
                                                                optimizer,   
                                                                criterion,  
                                                                num_epochs=num_epochs,   
                                                                use_patience=use_patience,   
                                                                patience=patience,  
                                                                model_dir=model_dir,   
                                                                model_name=model_name,   
                                                                save_model=save_model,  
                                                                batch_size=batch_size,
                                                                multi_objective=True if y_train.shape[1] > 1 else False
    )  

    # Evaluation on the test set (if test_size > 0.0)  
    if test_size > 0.0:  
        logging.info(f"Making predictions ...")  
        
        # Start timing for evaluation  
        wallclock_start = time.perf_counter()  
        cputime_start = time.process_time()  

        model.eval()  # Set to evaluation mode  
        with torch.no_grad():  # Disable gradient calculations  
            X_test = X_test.to(device)  # Ensure test set is on the same device  
            outputs = model(X_test)  # Get predictions for the test set  
            predictions = outputs.cpu().numpy()  # Convert predictions back to numpy array  
    
        # Stop timing for evaluation  
        prediction_wallclock = time.perf_counter() - wallclock_start   
        prediction_cputime = time.process_time() - cputime_start
        y_test = y_test.cpu()
    else:  
        predictions = None  
        prediction_wallclock, prediction_cputime = 0.0, 0.0  

    elapsed_time = {  
        "training_wallclock": training_wallclock,  
        "training_cputime": training_cputime,  
        "prediction_wallclock": prediction_wallclock,  
        "prediction_cputime": prediction_cputime  
    }  

    return y_test, predictions, history, elapsed_time  # Return predictions, metrics, and training history  


def predict_with_pretrained_model(
    data, 
    target_column=None, 
    hidden_size=64, 
    output_size=1,
    learning_rate=0.00001, 
    dropout_rate=0.5, 
    num_layers=12, 
    batch_size=16,
    use_patience=True, 
    patience=50, 
    num_epochs=1000, 
    test_size=0.9, 
    val_size=0.2,   
    random_state=42, 
    model_dir="models", 
    model_name="best_model.pth",  
    save_model=False, 
    fine_tuning=True, 
    freeze_up_to=6
):  
    """  
    Load a saved neural network model and use it to predict the runtime on new test data.  
    If fine-tuning is enabled, the data is split into train, validation, and test sets,  
    and the first layers of the model are frozen.  

    Parameters:  
        data (str): Path to the CSV file containing the new dataset.  
        target_column (str): Name of the column to use as the target (y).  
        hidden_size (int): Number of neurons in the hidden layers (must match the saved model).
        output_size (int): Number of outputs (e.g., 1 for one regression task).
        learning_rate (float): Learning rate for the optimizer.   
        dropout_rate (float): Dropout rate for regularization (must match the saved model).    
        num_layers (int): Number of hidden layers to include (>=3 for increased depth). 
        batch_size (int): Size of each batch for training and validation.  
        use_patience (bool): Whether to use early stopping.     
        patience (int): Patience for early stopping.   
        num_epochs (int): Maximum number of training epochs.   
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).  
        val_size (float): Proportion of the training set to include in the validation split (default: 0.2).  
        random_state (int): Random seed for reproducibility (default: 42).  
        model_dir (str): Directory where the saved model is stored.  
        model_name (str): Name of the saved model file.  
        save_model (bool): Save the best model if False.   
        fine_tuning (bool): Whether to fine-tune the model (default: True).   
        freeze_layers (int): Number of layers to freeze (default: 8).  

    Returns:  
        predictions (np.array): Predicted values for the test set. 
    """   
    logging.info(f"Predicting with pretrained model ...")  

    # Select device (GPU if available)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    logging.info(f"Using device: {device}")  

    # Prepare features (X) and target(s) (y)  
    if isinstance(target_column, str):  # If it's a single target column  
        if target_column not in data.columns:  
            logging.error(f"Target column '{target_column}' not found in the dataset.")  

        # Single target: `y` will be a 1D vector  
        X = data.drop(columns=[target_column]).values  # Drop the single target column  
        y = data[target_column].values  

    elif isinstance(target_column, list):  # If there are multiple target columns  
        for col in target_column:  
            if col not in data.columns:  
                logging.error(f"Target column '{col}' not found in the dataset.")  

        # Multiple targets: `y` will be a 2D matrix  
        X = data.drop(columns=target_column).values  # Drop all specified target columns  
        y = data[target_column].values  

    else:  # Raise an error if target_column is invalid  
        logging.error("'target_column' must be a string (for one target) or a list (for multiple targets).")  

    # Convert data to PyTorch tensors and move to the selected device  
    X = torch.tensor(X, dtype=torch.float32).to(device)  

    # If there is one target, `y` must be reshaped to match model expectations  
    if len(y.shape) == 1:  # Single target: Make `y` a 2D tensor with shape (num_samples, 1)  
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)  
    else:  # Multi-target: `y` already has the correct shape (num_samples, num_targets)  
        y = torch.tensor(y, dtype=torch.float32).to(device)  

    # Load the saved model  
    model_path = os.path.join(model_dir, model_name)  
    if not os.path.exists(model_path):  
        logging.error(f"Model file not found at {model_path}")  
        exit(1)

    # Initialize the model with the same architecture as during training and move to the right device  
    input_size = X.shape[1] 
    model = L2PModel(input_size, hidden_size, output_size, dropout_rate, num_layers).to(device) 

    model.load_state_dict(torch.load(model_path, map_location=device))  
    logging.info(f"Best model loaded from {os.path.join(model_dir, model_name)}")  

    validate_model_architecture(
        model, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_rate, 
        num_layers
    )  

    # Fine-tuning logic  
    if fine_tuning:  
        logging.info(f"Fine tuning the model ...")  

        # Split data into train, validation, and test sets  
        X_train, X_test, y_train, y_test = train_test_split(
                                                X, 
                                                y, 
                                                test_size=test_size, 
                                                random_state=random_state
                                            )  
        X_train, X_val, y_train, y_val = train_test_split(
                                            X_train, 
                                            y_train, 
                                            test_size=val_size, 
                                            random_state=random_state
                                        )  

        # Freeze the first layers
        freeze_layers(model, freeze_up_to=freeze_up_to)

        # Fine-tune the model  
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)  
        criterion = torch.nn.MSELoss()  

        # Train the model on GPU (or CPU if not available)  
        model, history, finetuning_wallclock, finetuning_cputime = train_in_batches(  
                                                                        model, 
                                                                        X_train, 
                                                                        y_train, 
                                                                        X_val, 
                                                                        y_val, 
                                                                        optimizer, 
                                                                        criterion,  
                                                                        num_epochs=num_epochs, 
                                                                        use_patience=use_patience, 
                                                                        patience=patience,  
                                                                        model_dir=model_dir, 
                                                                        model_name=model_name, 
                                                                        save_model=save_model,
                                                                        batch_size=batch_size,
                                                                        multi_objective=True if y_train.shape[1] > 1 else False
        )  
    else:  
        # Use the entire dataset for testing  
        X_test, y_test = X, y  
        history = None  
        train_wallclock, train_cputime = 0.0, 0.0  

    # Make predictions  
    logging.info(f"Making predictions ...")   

    # Start instrumentation  
    wallclock_start = time.perf_counter()  
    cputime_start = time.process_time()  

    model.eval()  
    with torch.no_grad():  
        outputs = model(X_test)  # Perform forward prediction  
        predictions = outputs.cpu().numpy()  # Move the outputs to CPU and convert to numpy  
    
    # Stop instrumentation  
    prediction_wallclock = time.perf_counter() - wallclock_start   
    prediction_cputime = time.process_time() - cputime_start 

    elapsed_time = {  
        "finetuning_wallclock": finetuning_wallclock,  
        "finetuning_cputime": finetuning_cputime,  
        "prediction_wallclock": prediction_wallclock,  
        "prediction_cputime": prediction_cputime  
    }  

    return y_test.cpu(), predictions, history, elapsed_time  # Return predictions and metrics  

