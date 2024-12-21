import torch
from torchvision import models as models
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.2),
            nn.Conv2d(16, 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.Dropout(0.2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 30 * 30, 32),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
            # nn.ELU() # enforce positive output
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layer(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # Fully connected layers
        x = self.fc_layer(x)
        return x
    
class CNNModel2(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel2, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, output_dim)
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # # Freeze all layers initially
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # # Unfreeze the first layer (initial conv layer)
        # for param in self.model.conv1.parameters():
        #     param.requires_grad = True

        # # Unfreeze the last fully connected layer
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class PercentageLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(PercentageLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # Ensure no division by zero by adding epsilon
        percentage_error = torch.abs((targets - predictions) / (targets + self.epsilon))
        return torch.mean(percentage_error) * 100  # Convert to percentage
