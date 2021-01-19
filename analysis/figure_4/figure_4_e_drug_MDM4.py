import pandas as pd
# from ggplot import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches

# def sigmoid(k,x,x0):
#     return (1 / (1 + np.exp(-k*(x-x0))))

def sigmoid(x, L ,k, x0):
    b=0.
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(5,4), dpi=200)


df = pd.read_excel('./CRISPR/092420 Data.xlsx', sheet_name='RO drug curves', header=[0,1], index_col=0)
# cols= ['RO5963'] + ['LNCaP']*6 +['PC3']*6 +['DU145']*6
cols= ['LNCaP']*6 +['PC3']*6 +['DU145']*6
# df = df[['LNCaP']*6]
print df.head()
# exps= df.columns.levels[0]
exps=['LNCaP', 'PC3', 'DU145']
print exps
# colors={'LNCaP':'maroon', 'PC3':'blue', 'DU145':'orange'}
colors={'LNCaP':'maroon', 'PC3':'#577399', 'DU145':'orange'}

X= df.index.values
legend_labels=[]
legnds=[]
for i, exp in enumerate(exps):
    print exp
    legend_labels.append(exp)
    df_exp = df[exp].copy()
    stdv = df_exp.std(axis=1)
    mean = df_exp.mean(axis=1)
    # ydata=mean.values
    # xdata = X
    ydata=df_exp.values.flatten()
    xdata = np.repeat(X, 6)
    # p0 = [max(ydata), np.median(xdata), -1.0, -0.5]  # this is an mandatory initial guess
    # p0 = [1.0, -1., -1.,-.7]  # initial guess
    p0 = [1.0, -1., -.7]  # initial guess
    # p0 = [2.0, 10., -.9]  # initial guess
    # p0 = [.5, 10., 0.0]  # initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0,method='dogbox', maxfev=60000)
    # popt, pcov = curve_fit(sigmoid, xdata, ydata,method='dogbox', maxfev=60000)
    print exp, popt
    # print popt, pcov
    #
    plt.errorbar(X,mean,yerr=stdv,fmt='o',ms=5,color=colors[exp],alpha=0.75,capsize=3, label=exp)
    x2= np.linspace((min(xdata), max(xdata)), 10)
    y2 = sigmoid( x2, *popt)
    plt.plot(x2,y2,color=colors[exp],alpha=0.75, linewidth=2)

    legnds.append(mpatches.Patch(color=colors[exp], label=exp))

plt.xscale('log')
plt.ylim((-.6, 1.6))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
plt.ylabel('Relative Viability',fontdict=dict(family='Arial', weight='bold', fontsize=14) )
plt.xlabel(u'\u03bcM RO-5963',fontdict=dict(family='Arial', weight='bold', fontsize=14) )
ax.spines['bottom'].set_position(('data',  0.))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
plt.xlim((.02,120))





# plt.legend(handles=legnds)

ax.legend( handles=legnds, bbox_to_anchor=(.9, 1.), framealpha=0.0)
# plt.legend()
# ax.legend(exps, fontsize=8, loc='upper right', framealpha=0.0)

plt.savefig('drug.png')



fig= plt.figure()
# x = np.linspace(-10, 10, 10)
# y1= sigmoid(x,L=1, x0=0, k=-1, b=0)
# y2= sigmoid(x,L=2, x0=5, k=-1, b=-0.5)
# y3= sigmoid(x,L=2, x0=5, k=-0.5, b=-0.5)
# popt1, pcov1 = curve_fit(sigmoid, x, y1, method='dogbox')
# popt2, pcov2 = curve_fit(sigmoid, x, y2, method='dogbox')
# popt3, pcov3 = curve_fit(sigmoid, x, y3, method='dogbox')
#
xfit = np.linspace(.025,120, 50)
# yfit1= sigmoid(xfit, *popt1)
# yfit2= sigmoid(xfit, *popt2)
# yfit3= sigmoid(xfit, *popt3)

# yfit1= sigmoid(xfit,  L=1, x0=0, k=-1, b=0 )
# yfit2= sigmoid(xfit,  L=2, x0=5, k=-1, b=-.5 )
# yfit3= sigmoid(xfit,  L=2, x0=5, k=-0.5, b=-.5 )
# yfit2= sigmoid(xfit, *popt2)
# yfit3= sigmoid(xfit, *popt3)

# y2= sigmoid(1,x, 1)
# y3= sigmoid(0.5,x, 0)
# plt.plot(xfit,yfit1, '-')
# plt.plot(xfit,yfit2, '-')
# plt.plot(xfit,yfit3, '-')
# #
# # plt.plot(x,y2, '*')
# # plt.plot(xfit,yfit2, '-')
#
# plt.plot(x,y3, '*')
# plt.plot(xfit,yfit3, '-')

# plt.plot(x,y3)
# plt.legend(['k=1, x0=0', 'k=1, x0=1', 'k=0.5, x0=0'])
plt.xscale('log')
plt.savefig('sigmoid.png')
# plt.legned(DU145='DU145', )

# cols= ['sgGFP','sgMDM4-1',	'sgMDM4-2']
# df = df[cols]
# df.columns= cols
# print df.head()
#
# print df.shape