import os
import re
import pandas as pd


# data_dir = os.path.dirname(__file__)
class GMT():
    # genes_cols : start reading genes from genes_col(default 1, it can be 2 e.g. if an information col is added after the pathway col)
    # pathway col is considered to be the first column (0)
    def load_data(self, filename, genes_col=1, pathway_col=0):

        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df

    def load_data_dict(self, filename):

        data_dict_list = []
        dict = {}
        with open(os.path.join(data_dir, filename)) as gmt:
            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.split('\t')
                dict[genes[0]] = genes[2:]

        return dict

    def write_dict_to_file(self, dict, filename):
        lines = []
        with open(filename, 'w') as gmt:
            for k in dict:
                str1 = '	'.join(str(e) for e in dict[k])
                line = str(k) + '	' + str1 + '\n'
                lines.append(line)
            gmt.writelines(lines)
        return

    def __init__(self):

        return
