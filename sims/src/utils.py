import torch
import os
import pandas as pd
import itertools

def get_param_dict():
    return {
            'macrophage_max_recruitment_rate': [0,8e-9],
            # 'macrophage_recruitment_min_signal': [0,0.2],
            # 'macrophage_recruitment_saturation_signal': [0,0.6],
            'DC_max_recruitment_rate': [0,4e-9],
            # 'DC_recruitment_min_signal': [0,0.2],
            # 'DC_recruitment_saturation_signal': [0,0.6],
            # 'DC_leave_rate': [0,0.4],
            # 'Th1_decay': [0,2.8e-6],
            # 'T_Cell_Recruitment': [0,2.2e-4],
            # 'DM_decay': [0,7e-4]
    }

def get_cell_df_columns():
    return [
				'position_x',
				'position_y',
				'cell_type',
				'total_volume',
				'current_phase',
				'nuclear_volume',
				# 'sensitivity_to_TNF_chemotaxis',
				# 'sensitivity_to_debris_chemotaxis',
				# 'debris_secretion_rate',
				# 'activated_TNF_secretion_rate',
				'activated_immune_cell'
				# 'TNF_decay_rate',
				# 'debris_decay_rate'
		]
def get_conc_df_columns():
    return [
                'mesh_center_m',
                'mesh_center_n',
                # 'mesh_center_o',
                'TNF',
                'debris'
        ]

def get_data_layers():
    return [
        # 'activated_TNF_secretion_rate',
        'activated_immune_cell',
        'cell_type_CD4_Tcell',
        'cell_type_CD8_Tcell',
        'cell_type_DC',
        'cell_type_cancer_cell',
        'cell_type_lung_cell',
        'cell_type_macrophage',
        'current_phase_G0G1_phase',
        'current_phase_G2_phase',
        'current_phase_M_phase',
        'current_phase_S_phase',
        'current_phase_apoptotic',
        # 'debris_secretion_rate',
        'nuclear_volume',
        # 'sensitivity_to_TNF_chemotaxis',
        # 'sensitivity_to_debris_chemotaxis',
        'total_volume',
        'TNF',
        'debris'
    ]

def get_param_ranges():
    params_dict = get_param_dict()
    return torch.tensor([x[1] - x[0] for x in params_dict.values()])

def get_system_env():
    return os.environ['SYSTEM_ENV']

def get_output_path():
    system_env = get_system_env()
    if system_env == 'laptop':
        return 'sims/dl_data/'
    else:
        return '/nfs/turbo/umms-ukarvind/joelne/mm_sims/recruit_rate/'

def expand_grid(**kwargs):
    """
    Create a DataFrame from all combinations of the supplied vectors.
    
    Parameters:
    **kwargs: Named vectors (lists, arrays, etc.)
    
    Returns:
    pd.DataFrame: A DataFrame containing all combinations of the inputs.
    """
    # Get the names and values of the input vectors
    keys = kwargs.keys()
    values = kwargs.values()
    
    # Use itertools.product to compute the Cartesian product
    combinations = list(itertools.product(*values))
    
    # Convert the combinations to a DataFrame
    return pd.DataFrame(combinations, columns=keys)
