import numpy as np
import pandas as pd
from src.utils import get_output_path
from src.data import make_tensor

OUTPUT_PATH = get_output_path()
# total_sims = 256
# sims_ids = range(total_sims)
sim_ids = [300]
grid_size = 64
timestep = 50

def main():
	# initialize the full array
	array_list = []
	for sim_id in sim_ids:
		print(sim_id)
		cell_df_path = OUTPUT_PATH + str(sim_id) + '_cells_df.parquet.gzip'
		conc_df_path = OUTPUT_PATH + str(sim_id) + '_conc_df.parquet.gzip'
		cell_df = pd.read_parquet(cell_df_path)
		conc_df = pd.read_parquet(conc_df_path)
		conc_df = conc_df.rename(columns={'mesh_center_m': 'position_x','mesh_center_n': 'position_y','mesh_center_o': 'position_z'})

		cell_tens, _ = make_tensor(cell_df,timestep,grid_size,'cell_df')
		conc_tens, _ = make_tensor(conc_df,timestep,grid_size,'conc_df')
		tens = np.concatenate((cell_tens,conc_tens),axis=2)
		tens = tens.transpose(2,0,1)
		array_list.append(tens)

	full_array = np.array(array_list)

	np.save(OUTPUT_PATH + str(timestep) + '_data.npy', full_array)

if __name__ == '__main__':
	main()
