# import matplotlib.pyplot as plt
# import numpy as np

# # fpr = np.loadtxt('./no_aug_fpr.txt')
# # tpr = np.loadtxt('./no_aug_tpr.txt')

# # fpr_20 = np.loadtxt('./20_aug_fpr.txt')
# # tpr_20 = np.loadtxt('./20_aug_tpr.txt')

# fpr_40_p = np.loadtxt('./no_aug_fpr.txt')
# tpr_40_p = np.loadtxt('./no_aug_tpr.txt')

# fpr_40 = np.loadtxt('./no_aug_fpr_1.txt')
# tpr_40 = np.loadtxt('./no_aug_tpr_1.txt')

# # fpr_60 = np.loadtxt('./60_aug_fpr.txt')
# # tpr_60 = np.loadtxt('./60_aug_tpr.txt')

# # fpr_80 = np.loadtxt('./80_aug_fpr.txt')
# # tpr_80 = np.loadtxt('./80_aug_tpr.txt')

# aoc_40_p = 0.76
# aoc_40 = 0.71

# plt.plot(fpr_40_p, tpr_40_p)
# plt.plot(fpr_40, tpr_40)

# plt.plot([0, 1], [0, 1], '--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend([f"40_aug_protected_words - {aoc_40_p} ", 
#             f"40_aug - {aoc_40}"], 
#            loc ="lower right")
# plt.show()

import pandas as pd

df = pd.read_csv('./data/dataset.csv')

print(df)

a1 = df['IMPORTANT '].values
a2 = df['IMPORTANT2'].values
a3 = df['IMPORTANT3'].values

a23 = 0

for x, y in zip(a2, a3):
    if(x == y):
        a23+=1

print(a23/len(a1)) 