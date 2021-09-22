from os.path import join

import pandas as pd

from config_path import PROSTATE_DATA_PATH

tmp_file = join(PROSTATE_DATA_PATH, 'raw_data/outputs_su2c_tcga_all_samples_n=980_rsem_results.tsv')
tpm_data = pd.read_csv(tmp_file, sep='\t', index_col=0)

mapping_ids_file = join(PROSTATE_DATA_PATH, 'raw_data/sample_mapping.tsv')
mapping_ids = pd.read_csv(mapping_ids_file, sep='\t')
tpm_id = mapping_ids[['tpm_col']].dropna()
tpm_id = tpm_id.set_index('tpm_col')
tpm_data['tpm_id'] = tpm_data['sample'].astype(str) + '_' + tpm_data['sequencing_type']

filtered_ge = tpm_data.join(tpm_id, how='inner', on='tpm_id')

filename = join(PROSTATE_DATA_PATH, 'processed/outputs_su2c_tcga_all_samples_n=980_rsem_results_filtered.tsv')
filtered_ge.to_csv(filename)
