from tqdm import tqdm
import torch
import torch.nn as nn
# from src.model import PercentageLoss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def training(model,train_loader,val_loader=None,epochs=100):
    validation_loss = []
    training_loss = []

    # Define loss and optimizer
    criterion = nn.MSELoss()  # Use a suitable loss function, e.g., MSE for regression
    # criterion = PercentageLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    early_stopping = EarlyStopping(patience=5, min_delta=0.0)

    for epoch in range(epochs):
        train_loss = 0.0
        num_batches = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            model.train()
            for input, target in tepoch:
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)

                # print(target[0])
                outputs = model(input)

                loss = criterion(outputs, target) 
                loss.backward()  # Backward pass
                optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                num_batches += 1
                
                # Update tqdm bar
                tepoch.set_postfix(batch_loss=loss.item())
            training_loss.append(train_loss / num_batches)

        if val_loader is not None:
            # Validation loop
            val_loss = 0.0
            num_batches = 0
            model.eval()
            with torch.no_grad():
                for input, target in val_loader:
                    input = input.to(device)
                    target = target.to(device)

                    output = model(input)

                    loss = criterion(output, target)
                    val_loss += loss.item()
                    num_batches += 1
            validation_loss.append(val_loss / num_batches)
            # Epoch-level metrics
            avg_val_loss = val_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average  Val Loss: {avg_val_loss:.4f}")
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    return model, training_loss, validation_loss
