# -*- coding: utf-8 -*-
"""
Here we partitioned the data into relevant confounding variables we are hoping
to analyze through chi square analysis
"""
import pandas as pd
import os
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn import preprocessing

df = pd.read_excel("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/Olympic-Analysis-master/Statiscal Analysis/multiple variables.xlsx")
m=df
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")


#get year vs. height vs. sport counts
hieghtvsportvyear = m.groupby(['Sport','Year','Height']).size().to_frame('count').reset_index()
hieghtvsportvyear.to_csv('Height_Sport_Year.csv')
df = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/Age_Sport_Year.csv")
multiindex_age = df.set_index(['Sport','Year','Age'])
multi_ind =multiindex_age.unstack()



#use dropna function to drop along certain axes
wrestles = multi_ind.loc["Archery"]
#drops all rows that contain only NAs
firstly = wrestles.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
#drops all columns that contain only NAs
secondly = firstly.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
endog=secondly
listof_indices = endog.index.tolist()
listof_columns = list(endog)

#graph functions Hodrick-Prescott (HP)
hp_cycle, hp_trend = sm.tsa.filters.hpfilter(endog, lamb=129600)
print("Hodrick-Prescott (HP) filter:\n")
print(hp_cycle)
print(hp_trend)
#plots the HP filter results
fig, axes = plt.subplots(2, figsize=(50,30));
axes[0].set(title='Level/trend component')
axes[0].plot(hp_trend, label='HP Filter')

axes[0].legend(loc='upper left')
axes[0].grid()

axes[1].set(title='Cycle component')
axes[1].plot(hp_cycle, label='HP Filter')
axes[1].legend(loc='upper left')
axes[1].grid()

fig.tight_layout();

#replce nan with 0 to ready dead for singular value decomposition/PCA
multi_ind2 = endog.where((pd.notnull(endog)),0)

#graph functions PCA
scaled_data = preprocessing.scale(multi_ind2)
pca =PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_*100,decimals =1)
labels =['PC' + str(x) for x in range(1,len(per_var)+1)]
#plots PCA as bar chart showing which PCs account for the most variance
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
#plots PCA as scatter plot to show sample deviance
pca_df = pd.DataFrame(pca_data,index=listof_indices, columns =labels)

plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1-{0}%'.format(per_var[0]))
plt.ylabel('PC2-{0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
plt.show()

#look at loading scores to identify samples that resulted in larged deviation along X-axis
loading_scores =pd.Series(pca.components_[0],index=listof_columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_years = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_years])




#Singular value decomposition
matrix_svd = linalg.svd(multi_ind2)

#get year vs. weight vs. sport counts
yearvweightvsport = m.groupby(['Sport','Year','Weight']).size().to_frame('count').reset_index()
yearvweightvsport.to_csv('Weight_Sport_Year.csv')


#get year vs. age vs. sport counts
yearvagevsport = m.groupby(['Sport','Year','Age']).size().to_frame('count').reset_index()
yearvagevsport.to_csv('Age_Sport_Year.csv')


#get country vs. sport counts
countryvsportvyear = m.groupby(['Sport','Year','NOC']).size().to_frame('count').reset_index()
countryvsportvyear.to_csv('Country_Sport_Year.csv')
