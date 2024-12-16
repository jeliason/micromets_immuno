import numpy as np
import pandas as pd

OUTPUT_PATH = "/nfs/turbo/umms-ukarvind/joelne/mm_sims/"
total_theta = 256
theta_ids = range(total_theta)
grid_size = 128
num_vars = 19
timestep = 100

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

	index = pd.MultiIndex.from_product([range(grid_size), range(grid_size), cols_for_full_grid], names=['grid_x', 'grid_y', 'variable'])
	full_grid = index.to_frame(index=False)
	full_grid = full_grid.merge(grid, on=['grid_x', 'grid_y', 'variable'],how='left').fillna(0)

	pivot = full_grid.set_index(['grid_x', 'grid_y', 'variable']).unstack(fill_value=0)

	tens = pivot.values.reshape(grid_size, grid_size, -1)

	return tens, sorted(full_grid['variable'].unique())

# initialize the full array
full_array = np.zeros((total_theta, grid_size, grid_size, num_vars))
for theta_id in theta_ids:
	print(theta_id)
	cell_df_path = OUTPUT_PATH + '/' + str(theta_id) + '_cells_df.parquet.gzip'
	conc_df_path = OUTPUT_PATH + '/' + str(theta_id) + '_conc_df.parquet.gzip'
	cell_df = pd.read_parquet(cell_df_path)
	conc_df = pd.read_parquet(conc_df_path)
	conc_df = conc_df.rename(columns={'mesh_center_m': 'position_x','mesh_center_n': 'position_y','mesh_center_o': 'position_z'})

	cell_tens, varnames = make_tensor(cell_df,timestep,grid_size,'cell_df')
	conc_tens, varnames = make_tensor(conc_df,timestep,grid_size,'conc_df')
	tens = np.concatenate((cell_tens,conc_tens),axis=2)
	full_array[theta_id,:,:,:] = tens

np.save(OUTPUT_PATH + str(timestep) + '_data.npy', full_array)
