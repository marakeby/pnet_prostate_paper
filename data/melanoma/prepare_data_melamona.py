import pandas as pd
from os.path import join
import sys
import os
from os.path import dirname, realpath, join
from config_path import MELANOMA_DATA_PATH, PLOTS_PATH


def prepare_design_matrix_crosstable( filter_silent_muts, filter_missense_muts, filter_introns_muts,
                                     keep_important_only=True, truncating_only=False):
    print('preparing mutations ...')
    id_col = 'Tumor_Sample_Barcode'

    mut_file = join(MELANOMA_DATA_PATH, 'raw_data/All_MutSig_QC_Pass_No_Novartis_UPDATED_Nov2019.maf')
    df = pd.read_csv(mut_file, sep='\t', low_memory=False)

    print ('mutation distribution')
    print df['Variant_Classification'].value_counts()

    ext = ""

    if filter_silent_muts:
        df = df[df['Variant_Classification'] != 'Silent'].copy()
        ext = "_no_silent"

    if filter_missense_muts:
        df = df[df['Variant_Classification'] != 'Missense_Mutation'].copy()
        ext = ext + "_no_missense"

    if filter_introns_muts:
        df = df[df['Variant_Classification'] != 'Intron'].copy()
        ext = ext + "_no_introns"

    # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
    exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
    if keep_important_only:
        df = df[~df['Variant_Classification'].isin(exclude)].copy()
        ext = 'important_only'
    if truncating_only:
        include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
        df = df[df['Variant_Classification'].isin(include)].copy()
        ext = 'truncating_only'
    # print df['Variant_Classification'].value_counts()
    df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                              aggfunc='count')
    df_table = df_table.fillna(0)
    total_numb_mutations = df_table.sum().sum()

    number_samples = df_table.shape[0]
    print 'number of mutations', total_numb_mutations, total_numb_mutations / (number_samples + 0.0)
    filename = join(MELANOMA_DATA_PATH, 'processed/M1000_final_analysis_set_cross_' + ext + '.csv')
    df_table.to_csv(filename)


def prepare_cnv():
    print('preparing copy number variants ...')
    filename = 'raw_data/cnv_all_thresholded.by_genes.txt'
    cnv_df = pd.read_csv(join(MELANOMA_DATA_PATH, filename), sep='\t', index_col=0)

    del cnv_df['Locus ID']
    del cnv_df['Cytoband']

    cnv_df = cnv_df.T
    cnv_df = cnv_df.fillna(0.)
    filename = join(MELANOMA_DATA_PATH, 'processed/M1000_data_CNA_paper.csv')
    cnv_df.to_csv(filename)
    print cnv_df.shape


def preapre_response():
    print('preparing copy number variants ...')
    filename = 'raw_data/clinical_annotation.txt'
    response_df = pd.read_csv(join(MELANOMA_DATA_PATH, filename), sep='\t', index_col=3)
    print response_df.head()
    filename = join(MELANOMA_DATA_PATH, 'processed/M1000_clinical.csv')
    response_df.to_csv(filename)


# remove silent and intron mutations
filter_silent_muts = False
filter_missense_muts = False
filter_introns_muts = False
keep_important_only = True
truncating_only = False

# prepare_design_matrix_crosstable(filter_silent_muts,filter_missense_muts, filter_introns_muts,keep_important_only=keep_important_only,truncating_only =truncating_only  )

# prepare_cnv()

preapre_response()