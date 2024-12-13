import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tempfile
import glob
import pcdl
import argparse
from scipy.stats import qmc

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

def generate_theta(n_samples,seed_start):
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

		for i,key in enumerate(params_dict.keys()):
				xml_root.find('.//user_parameters').find('.//' + key).text = str(params[i])

		tree.write(temp_path + '/PhysiCell_settings.xml')


def get_sim_df(temp_path):
		xmls = [file for file in glob.glob(temp_path + "/output*.xml")]
		cell_df_list = []
		conc_df_list  = []
		for i, xml in enumerate(sorted(xmls)):
				mcds = pcdl.TimeStep(xml)
				cell_df = mcds.get_cell_df()
				conc_df = mcds.get_conc_df()
				cell_df[["timestep"]] = i
				conc_df[["timestep"]] = i
				cell_df_list.append(cell_df)
				conc_df_list.append(conc_df)
		cells_df = pd.concat(cell_df_list,axis=0)
		conc_df = pd.concat(conc_df_list,axis=0)

		print("got sim dfs")
		return cells_df, conc_df


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

		# run simulations
		with tempfile.TemporaryDirectory() as temp_path:

				print("Starting config files...")
				config_files(temp_path,
										XML_FILE,
										params,
										seed)
				
				print("running physicell")
				subprocess.run(['./micromets_immuno',
												 temp_path + '/PhysiCell_settings.xml'],
												 stderr=subprocess.DEVNULL,
												 stdout=subprocess.DEVNULL)

				print("getting outputs")
				cells_df, conc_df = get_sim_df(temp_path)

		cells_df.to_parquet(OUTPUT_PATH + '/' + str(theta_id) + '_cells_df.parquet.gzip',index=False)
		conc_df.to_parquet(OUTPUT_PATH + '/' + str(theta_id) + '_conc_df.parquet.gzip',index=False)



if __name__ == "__main__":
		main()
