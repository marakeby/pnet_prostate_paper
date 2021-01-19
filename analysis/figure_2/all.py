from matplotlib import pyplot as plt, gridspec
import numpy as np

# from figure_2_a_mets_primary import plot_mets_primary
# from figure_2_b_auc_prc import plot_auc_all, plot_prc_all
# from figure_2_c_confusion_matrix import plot_confusion_matrix_all
from analysis.try_things.figure_2.figure2_c_gene_importance import plot_high_genes
from analysis.try_things.figure_2.figure2_d_activation import plot_activation
from figure_2_c_survival import plot_surv_all
from figure_2_b_external_validation import plot_external_validation_all, plot_external_validation_matrix
from figure_2_a_pnet_vs_dense import plot_pnet_vs_dense_auc, plot_pnet_vs_dense_ratio, plot_pnet_vs_dense_auc_with_ratio


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 3.0, 0.01)


fig= plt.figure(figsize=(13, 9), dpi=300)
# fig= plt.figure(figsize=(12, 8), dpi=600)
# fig= plt.figure(figsize=(15, 10), dpi=600)

gs = gridspec.GridSpec(ncols=7, nrows=4, figure=fig)

# gs = fig.add_gridspec(2,2)
# ax1 = fig.add_subplot(gs[0:2, :-1])
# ax2 = fig.add_subplot(gs[0, 2])
# ax22 = fig.add_subplot(gs[1,2])
# ax3 = fig.add_subplot(gs[2, 0])
# ax4 = fig.add_subplot(gs[2, 1])
# ax5 = fig.add_subplot(gs[2,2])
# plt.tight_layout(pad=3.0)

# ax000 = fig.add_subplot(gs[0, 0:2])
# ax001 = fig.add_subplot(gs[1, 0:2])
ax001 = fig.add_subplot(gs[0:2, 0:3])

ax02 = fig.add_subplot(gs[0:2,3:5])
ax03 = fig.add_subplot(gs[0:2, 5:7])

ax10 = fig.add_subplot(gs[2:4, 1:4])
ax11 = fig.add_subplot(gs[2:4,4:7])

plt.tight_layout(pad=3.0)
# plt.tight_layout()

# plt.subplots_adjust(bottom=0.1, right=0.9, left=0.1, top=0.9, wspace = 0.3, hspace = 0.3)
plt.subplots_adjust(bottom=0.07, right=0.99, left=0.09, top=0.95, wspace = 1.1, hspace = .7)
# plt.subplots_adjust(bottom=0.1, right=0.98, left=0.18, top=0.95)
# ax1 = plt.subplot(212 )
# ax1 = plt.subplot(231 )
# ax1.plot(t1, f(t1))

# plot_prc_all(ax1)
# ax2.margins(x=0, y=-0.25)-
# plot_confusion_matrix_all(ax2)
plot_surv_all(ax03)
# plot_external_validation_all(ax02)
plot_external_validation_matrix(ax02)
# ax02.margins(0.2)
# plot_pnet_vs_dense_auc(ax000)
# plot_pnet_vs_dense_ratio(ax001)
plot_pnet_vs_dense_auc_with_ratio(ax001)

plot_activation(ax10,column='coef_combined',  layer=3)
# ax10.margins(-.1)
# plot_high_genes(ax11, graph='viola', layer=1)
plot_high_genes(ax11, graph='swarm', layer=1)

# ax10.margins(2,0)           # Default margin is 0.05, value 0 means fit
# ax10.margins(x=0.1, y=0)   # Values in (-0.5, 0.0) zooms in to center


# ax2 = plt.subplot(221)
# ax2 = plt.subplot(233)
# ax2.margins(2, 2)           # Values >0.0 zoom out
# ax2.plot(t1, f(t1))
# plot_mets_primary(ax2)
# ax2.set_title('Zoomed out')

# ax3 = plt.subplot(222)
# ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
# # ax3.plot(t1, f(t1))
# plot_mets_primary(ax3)
# ax3.set_title('Zoomed in')

# plt.show()

# fig.tight_layout()
plt.savefig('./output/all.png')