import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import tempfile
import glob
import pcdl
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils import get_param_dict, get_output_path, get_cell_df_columns, get_conc_df_columns
import pandas as pd

XML_FILE = 'config/PhysiCell_settings.xml'

OUTPUT_PATH = get_output_path()

def config_files(temp_path,
								 xml_file,
								 params,
								 seed):
		tree = ET.parse(xml_file)
		xml_root = tree.getroot()
		xml_root.find('.//save').find('.//folder').text = temp_path + '/'

		xml_root.find('.//user_parameters').find('.//random_seed').text = str(seed)

		params_dict = get_param_dict()

		for i,key in enumerate(params_dict.keys()):
				
				xml_root.find('.//user_parameters').find('.//' + key).text = str(params[i])

		tree.write(temp_path + '/PhysiCell_settings.xml')

def write_sim_df(temp_path,cell_df_path,conc_df_path):
		cell_df_columns = get_cell_df_columns()
		conc_df_columns = get_conc_df_columns()
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

				table_cell = pa.Table.from_pandas(cell_df, preserve_index=False)
				table_conc = pa.Table.from_pandas(conc_df, preserve_index=False)

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
		parser.add_argument('--sim_id', type=int, default=0)

		args = parser.parse_args()
		sim_id = args.sim_id
		grid = pd.read_csv(OUTPUT_PATH + 'grid.csv')
		seed = sim_id
		
		theta_id = grid.loc[sim_id,'theta_id']
		theta = np.load(OUTPUT_PATH + 'theta.npy')
		params = theta[theta_id]

		print("thetas: ")
		print([f"{x:.2e}" for x in params])

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
				cell_df_path = OUTPUT_PATH + str(sim_id) + '_cells_df.parquet.gzip'
				conc_df_path = OUTPUT_PATH + str(sim_id) + '_conc_df.parquet.gzip'
				write_sim_df(temp_path,cell_df_path,conc_df_path)

		print("running make_array in R")
		subprocess.run(['Rscript','sims/make_tensor.R',str(sim_id)])



if __name__ == "__main__":
		main()
