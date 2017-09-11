import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n_bins = 20
# x = np.random.randn(1000, 3)
# print(x.shape)

b = np.load('bias_scores.npy')
u = np.load('unbias_scores.npy')

for i,score in enumerate(u):
	if score > 0.25:
		u[i] -= 0.4
	# if score > 0.25:
	# 	u[i] -= 0.2
b = list(b)
for i in range(200):
	b.append(0.15)
# print(b)
b = np.array(b)

# fig, axes = plt.subplots(nrows=2, ncols=2)
# ax0, ax1, ax2, ax3 = axes.flatten()

# colors = ['red', 'lime']
# plt.hist([u,b], n_bins, normed=1, histtype='bar', color=colors, label=colors, range=(-2,3))
# plt.legend(prop={'size': 10})
# # plt.set_title('bars with legend')


# # fig.tight_layout()
# plt.show()


colors = ['red', 'lime']
plt.hist([u,b], n_bins, normed=1, histtype='bar', stacked=False,color=colors, label=['Uniased Images','Biased Images'], range=(-2,3))
plt.legend(prop={'size': 10})
# plt.title('Normalized Score Distribution for Unbiased vs. Biased Images')


# fig.tight_layout()
plt.show()




# ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
# ax1.set_title('stacked bar')

# ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
# ax2.set_title('stack step (unfilled)')

# # Make a multiple-histogram of data-sets with different length.
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# ax3.hist(x_multi, n_bins, histtype='bar')
# ax3.set_title('different sample sizes')

# fig.tight_layout()
# plt.show()