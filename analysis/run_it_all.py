from matplotlib import pyplot as plt

from figure_1 import figure_1, figure_1_d_auc_prc, figure_1_e_confusion_matrix
from figure_2 import figure_2, figure_2_a_pnet_vs_dense, figure_2_b_external_validation, figure_2_c_survival
from figure_3 import prepare_data, figure_3_sankey, figure3_b_gene_importance, figure3_c_activation
from figure_4 import figure_4, figure_4_a, figure_4_d, figure_4_e_drug_MDM4, MDM4_TP53

# Figure 1
figure_1_d_auc_prc.run_auc()
plt.close()
figure_1_d_auc_prc.run_prc()
plt.close()
figure_1_e_confusion_matrix.run_matrix()
plt.close()
figure_1.run()
plt.close()

# # # # Figure 2
figure_2_a_pnet_vs_dense.run_pnet()
plt.close()
figure_2_b_external_validation.run_externnal_validation()
plt.close()
figure_2_c_survival.run()
figure_2.run()
plt.close()

# # Figure 3
prepare_data.run()
plt.close()
figure_3_sankey.run()
plt.close()
figure3_b_gene_importance.run()
plt.close()
figure3_c_activation.run()
plt.close()
#
# # # Figure 4
figure_4.run()
figure_4_a.run()
plt.clf()
figure_4_d.run()
plt.clf()
figure_4_e_drug_MDM4.run()
plt.clf()
MDM4_TP53.run()
plt.clf()

# Supp
# crossvalidation.run()
# plt.clf()
# layer_sizes.run()
# plt.clf()
# n_params.run()
# plt.clf()

## extended_figures
from extended_figures import figure_ed2_computational_performance, figure_ed3_pnet_vs_dense, figure_ed4_fusion, \
    figure_ed5_cnv, figure_ed5_importance, figure_ed7_activation

figure_ed2_computational_performance.run()
plt.clf()
figure_ed3_pnet_vs_dense.run()
plt.clf()
figure_ed4_fusion.run()
plt.clf()
figure_ed5_cnv.run()
plt.clf()
figure_ed5_importance.run()
plt.clf()
figure_ed7_activation.run()
plt.clf()
