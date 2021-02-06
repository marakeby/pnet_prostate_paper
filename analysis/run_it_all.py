from figure_1 import figure_1_d_auc_prc, figure_1_e_confusion_matrix
from figure_2 import figure_2_a_pnet_vs_dense, figure_2_b_external_validation, figure_2_c_survival
from figure_3 import prepare_data, figure_3_sankey_all,figure3_b_gene_importance, figure3_c_activation
from figure_4 import figure_4_a, figure_4_d, figure_4_e_drug_MDM4, MDM4_TP53
from sup import crossvalidation, layer_sizes, n_params

# Figure 1
figure_1_d_auc_prc.run_auc()
figure_1_d_auc_prc.run_prc()
figure_1_e_confusion_matrix.run_matrix()

# # # # Figure 2
figure_2_a_pnet_vs_dense.run_pnet()
figure_2_b_external_validation.run_externnal_validation()
figure_2_c_survival.run()

# # Figure 3
prepare_data.run()
figure_3_sankey_all.run()
figure3_b_gene_importance.run()
figure3_c_activation.run()

# # Figure 4
figure_4_a.run()
figure_4_d.run()
figure_4_e_drug_MDM4.run()
MDM4_TP53.run()

#Supp
crossvalidation.run()
layer_sizes.run()
n_params.run()

