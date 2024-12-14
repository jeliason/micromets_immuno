import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import glob
import pcdl
import argparse
from scipy.stats import qmc
import pyarrow.parquet as pq
import pyarrow as pa

XML_FILE = 'config/PhysiCell_settings.xml'
OUTPUT_PATH = '/nfs/turbo/umms-ukarvind/joelne/mm_sims'

params_dict = {
		'macrophage_max_recruitment_rate': [0,8e-9],
		'macrophage_recruitment_min_signal': [0,0.2],
		'macrophage_recruitment_saturation_signal': [0,0.6],
		'DC_max_recruitment_rate': [0,4e-9],
		'DC_recruitment_min_signal': [0,0.2],
		'DC_recruitment_saturation_signal': [0,0.6],
		'DC_leave_rate': [0,0.4],
		'Th1_decay': [0,2.8e-6],
		'T_Cell_Recruitment': [0,2.2e-4],
		'DM_decay': [0,7e-4]
}

def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0

def generate_theta(n_samples,seed_start):
		if not is_power_of_two(n_samples):
				raise ValueError("n_samples must be a power of 2")
		prior_min = [x[0] for x in params_dict.values()]
		prior_max = [x[1] for x in params_dict.values()]
		d = len(prior_min)
		sampler = qmc.Sobol(d=d, scramble=False, seed = seed_start)
		sample = sampler.random_base2(m=int(np.log2(n_samples)))
		sample = qmc.scale(sample,prior_min, prior_max)

		return sample

def config_files(temp_path,
								 xml_file,
								 params,
								 seed):
		tree = ET.parse(xml_file)
		xml_root = tree.getroot()
		xml_root.find('.//save').find('.//folder').text = temp_path + '/'

		xml_root.find('.//user_parameters').find('.//random_seed').text = str(seed)

		if params[1] > params[2]:
				params[1], params[2] = params[2], params[1]
		
		if params[4] > params[5]:
				params[4], params[5] = params[5], params[4]

		for i,key in enumerate(params_dict.keys()):
				
				xml_root.find('.//user_parameters').find('.//' + key).text = str(params[i])

		tree.write(temp_path + '/PhysiCell_settings.xml')


# def get_sim_df(temp_path):
# 		xmls = [file for file in glob.glob(temp_path + "/output*.xml")]
# 		cell_df_list = []
# 		conc_df_list  = []
# 		for i, xml in enumerate(sorted(xmls)):
# 				mcds = pcdl.TimeStep(xml,graph=False)
# 				cell_df = mcds.get_cell_df()
# 				conc_df = mcds.get_conc_df()
# 				cell_df[["timestep"]] = i
# 				conc_df[["timestep"]] = i
# 				cell_df_list.append(cell_df)
# 				conc_df_list.append(conc_df)
# 		cells_df = pd.concat(cell_df_list,axis=0)
# 		conc_df = pd.concat(conc_df_list,axis=0)

# 		print("got sim dfs")
# 		return cells_df, conc_df

def write_sim_df(temp_path,cell_df_path,conc_df_path):
    cell_df_columns = [
        'position_x',
        'position_y',
        'cell_type',
        'total_volume',
        'current_phase',
        'nuclear_volume',
        'sensitivity_to_TNF_chemotaxis',
        'sensitivity_to_debris_chemotaxis',
        'debris_secretion_rate',
        'activated_TNF_secretion_rate',
        'activated_immune_cell'
        # 'TNF_decay_rate',
        # 'debris_decay_rate'
    ]
    conc_df_columns = [
        'mesh_center_m',
        'mesh_center_n',
        'TNF',
        'debris'
    ]
    xmls = [file for file in glob.glob(temp_path + "/output*.xml")]
    writer_cell_df = None
    writer_conc_df = None
    compression_codec = "snappy"
    for i, xml in enumerate(sorted(xmls)):
        if i < 12:
            continue
        mcds = pcdl.TimeStep(xml,graph=False)
        cell_df = mcds.get_cell_df()
        conc_df = mcds.get_conc_df()

        cell_df = cell_df[cell_df_columns]
        conc_df = conc_df[conc_df_columns]
        
        cell_df[["timestep"]] = i
        conc_df[["timestep"]] = i

        table_cell = pa.Table.from_pandas(cell_df)
        table_conc = pa.Table.from_pandas(conc_df)

        if writer_cell_df is None:
            writer_cell_df = pq.ParquetWriter(cell_df_path, table_cell.schema, compression=compression_codec)
            writer_conc_df = pq.ParquetWriter(conc_df_path, table_conc.schema, compression=compression_codec)

        writer_cell_df.write_table(table_cell)
        writer_conc_df.write_table(table_conc)

    if writer_cell_df is not None:
        writer_cell_df.close()
        writer_conc_df.close()


    print("wrote sim dfs")


def main():
		parser = argparse.ArgumentParser()
		parser.add_argument('--theta_id', type=int, default=0)
		parser.add_argument('--n_samples', type=int, default=1000)
		parser.add_argument('--seed_start', type=int, default=1234)

		args = parser.parse_args()
		theta_id = args.theta_id
		seed_start = args.seed_start
		n_samples = args.n_samples

		seed = seed_start + theta_id
		
		# generate theta
		theta = generate_theta(n_samples,seed_start)
		params = theta[theta_id,]

		print("thetas: ")
		print(params)

		# run simulations
		with tempfile.TemporaryDirectory() as temp_path:

				print("Starting config files...")
				config_files(temp_path,
										XML_FILE,
										params,
										seed)
				
				print("running physicell")
				subprocess.run(['./micromets_immuno',
												 temp_path + '/PhysiCell_settings.xml']
												)

				print("getting outputs")
				cell_df_path = OUTPUT_PATH + '/' + str(theta_id) + '_cells_df.parquet.gzip'
				conc_df_path = OUTPUT_PATH + '/' + str(theta_id) + '_conc_df.parquet.gzip'
				write_sim_df(temp_path,cell_df_path,conc_df_path)

		# cells_df.to_parquet(OUTPUT_PATH + '/' + str(theta_id) + '_cells_df.parquet.gzip',index=False)
		# conc_df.to_parquet(OUTPUT_PATH + '/' + str(theta_id) + '_conc_df.parquet.gzip',index=False)



if __name__ == "__main__":
		main()
