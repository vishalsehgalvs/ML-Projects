# ============================================================
# DIMENSIONALITY REDUCTION — Principle Component Analysis
# ============================================================

# importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X,y = make_blobs(n_samples=500,n_features=5,centers=3,cluster_std=1.5,random_state=42)
# print(X)
# [[ -9.85712583   9.52196609   6.40680626  -6.81757623  -7.86054541]
#  [ -8.04717781   8.40261648   6.40946097  -4.33576029  -6.70289196]
#  [ -3.73690895   6.7601386    4.24877609   0.28504117  -7.10318219]
#  ...
#  [ -5.82805     -7.29173339   7.48633693   2.71623974   7.33823548]
#  [ -2.48648271   9.67739715   6.21190845   1.03031506  -7.54637531]
#  [ -8.63901448 -10.68852991   8.36601977   4.94211449   3.77070371]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled)
# [[-1.07039133  0.74546814  0.0859273  -1.58466105 -0.90798501]
#  [-0.52251206  0.61451192  0.08731966 -0.95761934 -0.68801347]
#  [ 0.78222538  0.4223533  -1.04593191  0.20984645 -0.76407462]
#  ...
#  [ 0.14922774 -1.22161917  0.65212732  0.82409945  1.98001159]
#  [ 1.16073493  0.7636525  -0.01629411  0.39814317 -0.84828797]
#  [-0.70166364 -1.61902103  1.11350971  1.38647649  1.30212699]]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# print(X_pca)
# [[ 1.40522818e+00 -1.73079914e+00]
#  [ 1.03463536e+00 -9.82792414e-01]
#  [ 1.10182830e+00  1.09084050e+00]
# .......500

df_pca = pd.DataFrame(X_pca,columns=['PC1','PC2'])
df_pca['label'] = y

sns.scatterplot(data=df_pca,x='PC1',y='PC2',hue = 'label',palette = 'Set2')
plt.show()









