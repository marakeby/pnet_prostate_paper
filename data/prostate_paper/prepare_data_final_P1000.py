import pandas as pd
from os.path import join
from config_path import PROSTATE_DATA_PATH

processed_dir = 'processed'
data_dir = 'raw_data'

processed_dir = join(PROSTATE_DATA_PATH, processed_dir)
data_dir = join(PROSTATE_DATA_PATH, data_dir)


def prepare_design_matrix_crosstable():
    print('preparing mutations ...')

    filename = '41588_2018_78_MOESM4_ESM.txt'
    id_col = 'Tumor_Sample_Barcode'
    df = pd.read_csv(join(data_dir, filename), sep='\t', low_memory=False, skiprows=1)
    print ('mutation distribution')
    print df['Variant_Classification'].value_counts()

    if filter_silent_muts:
        df = df[df['Variant_Classification'] != 'Silent'].copy()
    if filter_missense_muts:
        df = df[df['Variant_Classification'] != 'Missense_Mutation'].copy()
    if filter_introns_muts:
        df = df[df['Variant_Classification'] != 'Intron'].copy()

    # important_only = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Splice_Site','Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Start_Codon_SNP','Nonstop_Mutation', 'De_novo_Start_OutOfFrame', 'De_novo_Start_InFrame']
    exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'lincRNA']
    if keep_important_only:
        df = df[~df['Variant_Classification'].isin(exclude)].copy()
    if truncating_only:
        include = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']
        df = df[df['Variant_Classification'].isin(include)].copy()
    df_table = pd.pivot_table(data=df, index=id_col, columns='Hugo_Symbol', values='Variant_Classification',
                              aggfunc='count')
    df_table = df_table.fillna(0)
    total_numb_mutations = df_table.sum().sum()

    number_samples = df_table.shape[0]
    print 'number of mutations', total_numb_mutations, total_numb_mutations / (number_samples + 0.0)
    filename = join(processed_dir, 'P1000_final_analysis_set_cross_' + ext + '.csv')
    df_table.to_csv(filename)


def prepare_response():
    print('preparing response ...')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(join(data_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
    response = pd.DataFrame()
    response['id'] = df['Patient.ID']
    response['response'] = df['Sample.Type']
    response['response'] = response['response'].replace('Metastasis', 1)
    response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    response.to_csv(join(processed_dir, 'response_paper.csv'), index=False)


def prepare_cnv():
    print('preparing copy number variants ...')
    filename = '41588_2018_78_MOESM10_ESM.txt'
    df = pd.read_csv(join(data_dir, filename), sep='\t', low_memory=False, skiprows=1, index_col=0)
    df = df.T
    df = df.fillna(0.)
    filename = join(processed_dir, 'P1000_data_CNA_paper.csv')
    df.to_csv(filename)


def prepare_cnv_burden():
    print('preparing copy number burden ...')
    filename = '41588_2018_78_MOESM5_ESM.xlsx'
    df = pd.read_excel(join(data_dir, filename), skiprows=2, index_col=1)
    cnv = df['Fraction of genome altered']
    filename = join(processed_dir, 'P1000_data_CNA_burden.csv')
    cnv.to_frame().to_csv(filename)


# remove silent and intron mutations
filter_silent_muts = False
filter_missense_muts = False
filter_introns_muts = False
keep_important_only = True
truncating_only = False

ext = ""
if keep_important_only:
    ext = 'important_only'

if truncating_only:
    ext = 'truncating_only'

if filter_silent_muts:
    ext = "_no_silent"

if filter_missense_muts:
    ext = ext + "_no_missense"

if filter_introns_muts:
    ext = ext + "_no_introns"

prepare_design_matrix_crosstable()
prepare_cnv()
prepare_response()
prepare_cnv_burden()
print('Done')
