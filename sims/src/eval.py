import torch
import torch.nn as nn
from src.utils import get_param_ranges, get_param_dict
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
from src.data import inverse_transform

def make_predictions(model,newdata_loader,Theta=None):
	model.eval()

	# Prepare to store predictions and actual values
	predictions_list = []
	actuals_list = []

	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	model.to(device)

	with torch.no_grad():
			for inputs, targets in newdata_loader:
					inputs = inputs.to(device)
					targets = targets.to(device)
					outputs = model(inputs)

					# Denormalize the outputs
					predictions = inverse_transform(outputs, Theta)
					targets = inverse_transform(targets, Theta)
					# predictions = outputs * param_ranges
					# print(predictions)
					# Collect predictions and actuals
					predictions_list.extend(predictions.tolist())  # List of vectors
					actuals_list.extend(targets.tolist())

	return predictions_list, actuals_list

def predictions_table(model, val_loader,Theta=None):
	# Pretty print results in a stacked format
	params_dict = get_param_dict()
	nms = list(params_dict.keys())
	table = PrettyTable(["Index", "Type"] + nms)
	percent_diffs = []
	predictions_list, actuals_list = make_predictions(model, val_loader,Theta)
	for idx, (pred_vec, actual_vec) in enumerate(zip(predictions_list, actuals_list)):
			# Format predictions and actuals as strings
			# percent difference compared to actual
			pred_vec = np.array(pred_vec)
			actual_vec = np.array(actual_vec)
			percent_diff = np.abs((pred_vec - actual_vec) / (actual_vec + 1e-16)) * 100
			percent_diffs.append(percent_diff)
			# Add rows for predictions and actuals
			table.add_row([idx, "Prediction"] + [f"{x:.2e}" for x in pred_vec])
			table.add_row([idx, "Actual"] + [f"{x:.2e}" for x in actual_vec])
			table.add_row([idx, "Percent Difference"] + [f"{x:.2f}%" for x in percent_diff])
			table.add_row(["-"*5, "-"*9] + ["-"*10 for _ in range(len(nms))])  # Separator for better readability

	meds = np.median(np.array(percent_diffs),axis=0)
	q10 = np.percentile(np.array(percent_diffs),10,axis=0)
	q90 = np.percentile(np.array(percent_diffs),90,axis=0)
	table.add_row(["10th Percentile", "Percent Difference"] + [f"{x:.2f}%" for x in q10])
	table.add_row(["Median", "Percent Difference"] + [f"{x:.2f}%" for x in meds])
	table.add_row(["90th Percentile", "Percent Difference"] + [f"{x:.2f}%" for x in q90])
	print(table)


def plot_learning_curve(training_loss, validation_loss):
	epochs = range(1, len(training_loss) + 1)

	plt.figure(figsize=(10, 6))
	plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
	plt.plot(epochs, validation_loss, 'ro-', label='Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.show()
