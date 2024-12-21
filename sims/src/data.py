from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import get_param_ranges, get_data_layers
import pandas as pd


class ImageDataset(Dataset):
		def __init__(self, data, targets, transform=None,target_transform=None):
				self.data = data
				self.targets = targets
				self.transform = transform
				self.target_transform = target_transform

		def __len__(self):
				return len(self.data)

		def __getitem__(self, idx):
				x = self.data[idx]
				y = self.targets[idx]
				if self.transform:
						x = self.transform(x)
				if self.target_transform:
						y = self.target_transform(y)
				return x, y
		
# Shuffled-label dataset wrapper
class ShuffledLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Shuffle the labels
        self.shuffled_idx = torch.randperm(len(dataset))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore original label
        _, shuffled_label = self.dataset[self.shuffled_idx[idx]]
        return data, shuffled_label
		
def make_target_transform(Theta=None):
	param_ranges = get_param_ranges()
	# log_targets = torch.log1p(Theta)

	# # Compute the mean and standard deviation
	# log_mean = log_targets.mean()
	# log_std = log_targets.std()

	target_transform = transforms.Compose([
		transforms.Lambda(lambda x: x / param_ranges), # Normalize to range [0, 1]
		transforms.Lambda(lambda x: x * 2 - 1),  # Normalize to range [-1, 1]
		# transforms.Lambda(lambda x: torch.log1p(x)),
		# transforms.Lambda(lambda x: (x - log_mean) / log_std)
	])

	return target_transform

def inverse_transform(y,Theta=None):
	param_ranges = get_param_ranges()
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	param_ranges = param_ranges.to(device)
	# log_targets = torch.log1p(Theta)

	# # Compute the mean and standard deviation
	# log_mean = log_targets.mean()
	# log_std = log_targets.std()

	# # Undo the normalization
	# y = y * log_std + log_mean
	# y = torch.expm1(y)
	# y = y * param_ranges
	y = (y + 1) / 2	# Denormalize to range [0, 1]
	y = y * param_ranges

	return y
		

def make_loader(data,Theta=None,get_shuffled=False):
		transform = transforms.Compose([
			# transforms.ToTensor(),  # Convert image to tensor
			# transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to range [-1, 1]
			transforms.RandomRotation(degrees=(0,360)),  # Rotate randomly within ±30 degrees
			transforms.RandomHorizontalFlip(p=0.75),  # Flip horizontally with 50% probability
			transforms.RandomVerticalFlip(p=0.75),  # Flip vertically with 50% probability
			transforms.RandomResizedCrop(size=(128,128), scale=(0.5, 1.0),ratio=(3/4,4/3),antialias=True),  # Random crop and resize
		])

		target_transform = make_target_transform(Theta)

		dataset = ImageDataset(data, Theta, transform=transform,target_transform=target_transform)

		train_size = int(0.8 * len(dataset))
		val_size = len(dataset) - train_size

		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

		batch_size = 32

		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

		# Verify the split
		print(f"Training dataset size: {len(train_dataset)}")
		print(f"Validation dataset size: {len(val_dataset)}")

		if get_shuffled:
			return train_loader, val_loader, DataLoader(ShuffledLabelDataset(train_dataset), batch_size=batch_size, shuffle=True)
		else:
			return train_loader, val_loader

def plot_data(data_slice):

	nms = get_data_layers()

	vmin = 0
	# Define the number of rows and columns for the facet grid
	n_slices = data_slice.shape[0]
	n_cols = 3  # Adjust based on your preference
	n_rows = (n_slices + n_cols - 1) // n_cols  # Calculate rows to fit all slices

	# Create a figure with subplots
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

	# Flatten the axes array for easy iteration (handle case where there's only one row)
	axes = axes.flatten()

	# Iterate through slices and plot
	for i in range(n_slices):
			sns.heatmap(data_slice[i, :, :], ax=axes[i], cbar=True, cmap="magma_r", vmin=vmin)
			axes[i].set_title(nms[i])

	# Hide any unused subplots
	for j in range(n_slices, len(axes)):
			axes[j].axis('off')

	# Adjust layout for readability
	plt.tight_layout()
	plt.show()

def make_tensor(sims_df,timestep,grid_size,type=None,x_range=(-400,400),y_range=(-400,400)):
	df = sims_df[sims_df['timestep'] == timestep].copy()
	if type == 'cell_df':
		cell_types = sorted([
			'lung_cell',
			'cancer_cell',
			'CD8_Tcell',
			'macrophage',
			'DC',
			'CD4_Tcell'
		])
		phases = sorted([
			'G0G1_phase',
			'G2_phase',
			'S_phase',
			'M_phase',
			'apoptotic'
		])
		df['cell_type'] = pd.Categorical(df['cell_type'], categories=cell_types)
		df['current_phase'] = pd.Categorical(df['current_phase'], categories=phases)
		df = pd.get_dummies(df, columns=['current_phase','cell_type'])

	x_min, x_max = x_range
	y_min, y_max = y_range
	x_step = (x_max - x_min) / grid_size
	y_step = (y_max - y_min) / grid_size

	# Assign cells to grid pixels
	df['grid_x'] = ((df['position_x'] - x_min) // x_step).astype(int)
	df['grid_y'] = ((df['position_y'] - y_min) // y_step).astype(int)

	# Ensure grid_x and grid_y are within bounds
	df['grid_x'] = df['grid_x'].clip(0, grid_size - 1)
	df['grid_y'] = df['grid_y'].clip(0, grid_size - 1)

	df = df.drop(columns=['position_x', 'position_y', 'timestep'])
	cols_for_full_grid = df.drop(columns=['grid_x', 'grid_y']).columns

	# Group by grid and cell type, and then sum the continuous variables
	grid = (
			df.melt(id_vars=['grid_x', 'grid_y'], var_name='variable', value_name='value')
			.groupby(['grid_x', 'grid_y','variable'])
			['value']
			.sum()
			.reset_index()
	)

	index = pd.MultiIndex.from_product([range(grid_size), range(grid_size), cols_for_full_grid], names=['grid_y', 'grid_x', 'variable'])
	full_grid = index.to_frame(index=False)
	full_grid = full_grid.merge(grid, on=['grid_y', 'grid_x', 'variable'],how='left').fillna(0)

	pivot = full_grid.set_index(['grid_y', 'grid_x', 'variable']).unstack(fill_value=0)

	tens = pivot.values.reshape(grid_size, grid_size, -1)

	return tens, sorted(full_grid['variable'].unique())
