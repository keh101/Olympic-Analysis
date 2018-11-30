# -*- coding: utf-8 -*-
"""
Here we decomposed the variables by PCA and HO Filter analysis to look for trends in the data set
"""
import pandas as pd
import os
import csv
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
params = {'legend.fontsize': 30, 'legend.handlelength': 3}
plt.rcParams.update(params)

from scipy import linalg
from sklearn.decomposition import PCA
from sklearn import preprocessing


def prepare_data(multiindex,Sport, nan='yes'):
    """
    This takes in multiindexed data and prepaes the data for analysis by each specific Sport name typed in.
    It returns the prepared data, a list of final indices, and list of column names in a dictionary
    """
    assert nan=='yes' or nan =='no'
    assert isinstance(Sport,str)
    
    unnitsport = multiindex.index.tolist()
    notempt = [item for item in unnitsport if Sport in item]
    if not notempt:
        print ("The sport %s is not in the indices" % Sport)
        return None
    else:
        multi_ind =multiindex.unstack()
        #use dropna function to drop along certain axes
        wrestles = multi_ind.loc[Sport]
        #drops all rows that contain only NAs
        firstly = wrestles.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
        #drops all columns that contain only NAs
        secondly = firstly.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
        #replce nan with 0 to ready dead for singular value decomposition/PCA
        multi_ind2 = secondly.where((pd.notnull(secondly)),0)
        print(multi_ind2)
        #put index and column names into lists
        listof_indices = secondly.index.tolist()
        listof_columns = list(secondly)
        #check to make sure there are enough samples
        if len(listof_indices) >0:
            #return nan containing OR 0 containing prepared_data 
            if nan == 'yes':
                return{'listof_indices':listof_indices,'listof_columns':listof_columns,'prepared_data':secondly}
            if nan == 'no':
                return{'listof_indices':listof_indices,'listof_columns':listof_columns,'prepared_data':multi_ind2}
            print('Data for %s prepared successfully'%Sport)
        else:
            print("The sport, %s , has 0 valid samples" % Sport)
            
def calculate_hp_trends(endog):
    """
    calculated HP trend arrays
    """
    #creates array of Hodrick-Prescott (HP) trends and cycles
    hp_cycle, hp_trend = sm.tsa.filters.hpfilter(endog, lamb=129600)
    print("Hodrick-Prescott (HP) filter calculated:\n")
    return {'hp_cycle':hp_cycle,'hp_trend':hp_trend}

def graph_hp_trends(Sport,hp_cycle,hp_trend,variable_of_int,path):
    """
    Graphs HP trend arrays
    """
    #plots the HP filter results
    firstly = hp_trend.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    #drops all columns that contain only NAs
    secondly = firstly.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
    hp_trend = secondly
    
    #plots the HP filter results
    firstly = hp_cycle.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    #drops all columns that contain only NAs
    secondly = firstly.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
    hp_cycle = secondly
    
    if not hp_trend.empty:
        hp_trend_graph = hp_trend['count'].plot(figsize=(50,25), fontsize =40);
        hp_trend_graph.set_title('HP Trend-{0}'.format(Sport),fontsize =60)
        hp_trend_graph.set_xlabel('Year', fontsize =40)
        hp_trend_graph.set_ylabel('Tau(t)', fontsize =40)
        hp_trend_graph.get_legend().get_title().set_fontsize(30)
        fig1 = hp_trend_graph.get_figure()
        fig1.savefig(r'%s\Olympic-Analysis\AnalyzedData\%s\Figures\%s_hptrends.png' %(path,variable_of_int,Sport))
    else:
        print('There were no trends in %s vs. %s'%(Sport,variable_of_int))

    if not hp_cycle.empty:
        hp_cycle_graph = hp_cycle['count'].plot(figsize=(50,25),fontsize =40);
        hp_cycle_graph.set_title('HP Cycle-{0}'.format(Sport),fontsize =60)
        hp_cycle_graph.set_xlabel('Year', fontsize =40)
        hp_cycle_graph.set_ylabel('C(t)', fontsize =40)
        hp_cycle_graph.get_legend().get_title().set_fontsize(30)
        fig2 = hp_cycle_graph.get_figure()
        fig2.savefig(r'%s\Olympic-Analysis\AnalyzedData\%s\Figures\%s_hpcycles.png' %(path,variable_of_int,Sport))
    else:
        print('There were no cycles in %s vs. %s'%(Sport,variable_of_int))
        
def calculate_PCA(prepared_data):
    """
    calculates PCA data 
    """
    #graph functions PCA
    scaled_data = preprocessing.scale(prepared_data)
    pca =PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_*100,decimals =1)
    labels =['PC' + str(x) for x in range(1,len(per_var)+1)]
    return{'pca':pca,'pca_data':pca_data,'per_var':per_var,'labels':labels}

def plot_bar_PCA(per_var,labels,Sport,variable_of_int,path):
    """
    Plots PCA bar graph data showing which PC's contribute to the most deviation
    """
    #plots PCA as bar chart showing which PCs account for the most variance
    plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(r'%s\Olympic-Analysis\AnalyzedData\%s\PCA\%s_PCA_Bar.png' %(path,variable_of_int,Sport))
    plt.show()

def plot_scatter_PCA(pca_data,listof_indices,labels,per_var,Sport, variable_of_int,path):
    """
    Plots a scatter plot of the PCA data for each Sport. Plots by year.
    """
    #plots PCA as scatter plot to show sample deviance
    pca_df = pd.DataFrame(pca_data,index=listof_indices, columns =labels)
    
    fig2 = plt.scatter(pca_df.PC1,pca_df.PC2)
    plt.title('%s PCA Graph'%Sport)
    plt.xlabel('PC1-{0}%'.format(per_var[0]))
    plt.ylabel('PC2-{0}%'.format(per_var[1]))
    
    for sample in pca_df.index:
        plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
    plt.show()
    fig2.get_figure().savefig(r'%s\Olympic-Analysis\AnalyzedData\%s\PCA\%s_PCA_Scatter.png' %(path,variable_of_int,Sport))
    
def top_10_PCA(pca,listof_columns,Sport,variable_of_int,path):
    """
    Returns and saves a list of the top 10 contributing factors causing deviations to a csv
    """
    #look at loading scores to identify samples that resulted in larged deviation along X-axis
    loading_scores =pd.Series(pca.components_[0],index=listof_columns)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_10_years = sorted_loading_scores[0:10].index.values
    loading_scores[top_10_years].to_csv(r'%s\Olympic-Analysis\AnalyzedData\%s\PCA\%s_top10_PCA.csv' %(path,variable_of_int,Sport))
    return loading_scores[top_10_years]

def perform_trend_analysis(variable_of_int,path):
    """
    Calls HP filter functions
    """
    #read in sports as a list
    os.chdir(r'%s\Olympic-Analysis\Olympic-Analysis-master\120-years-of-olympic-history-athletes-and-results'%path)
    with open('sports.csv', 'r') as f:
        reader = csv.reader(f)
        listofsports = list(reader)
    #read in data
    df_to = pd.read_csv(r"%s/Olympic-Analysis/AnalyzedData/%s_Sport_Year.csv"%(path,variable_of_int))
    multiindex = df_to.set_index(['Sport','Year',variable_of_int])
    print(multiindex)
    
 #   Calculate and graph HP data for variable_of_int
    for sports in listofsports:
        print(sports)
        dict_of_nan = prepare_data(multiindex,sports[0], nan='yes')
        try:    
            if dict_of_nan is not None:
                hp_info = calculate_hp_trends(dict_of_nan['prepared_data'])
                graph_hp_trends(sports[0],hp_info['hp_cycle'],hp_info['hp_trend'],variable_of_int,path)
        except ValueError:
            print("the sport %s caused a value error, most liekly due to nonmatching shape of passed values"%sports)
        
        print('next_sport')


def perform_PCA_analysis(variable_of_int,path):
    """
    Calls PCA functions
    """
    #read in sports as a list
    os.chdir(r'%s\Olympic-Analysis\Olympic-Analysis-master\120-years-of-olympic-history-athletes-and-results'%path)
    with open('sports.csv', 'r') as f:
        reader = csv.reader(f)
        listofsports = list(reader)
    #read in data
    df_to = pd.read_csv(r"%s/Olympic-Analysis/AnalyzedData/%s_Sport_Year.csv"%(path,variable_of_int))
    df_to = df_to.drop(['Unnamed: 0'], axis=1)
    print(df_to.head(n=5))
    multiindex = df_to.set_index(['Sport','Year',variable_of_int])
    print(multiindex.head(n=5))
    #Calculate and graph HP data for variable_of_int
    for sports in listofsports[63:]:
        try:
            print(sports)
            dict_of_0 = prepare_data(multiindex,sports[0], nan='no')
            if dict_of_0 is not None:
                    pca_info = calculate_PCA(dict_of_0['prepared_data'])
                    plot_bar_PCA(pca_info['per_var'],pca_info['labels'],sports[0],variable_of_int,path)
                    plot_scatter_PCA(pca_info['pca_data'],dict_of_0['listof_indices'],pca_info['labels'],pca_info['per_var'],sports[0],variable_of_int,path)
                    top10_info = top_10_PCA(pca_info['pca'],dict_of_0['listof_columns'],sports[0],variable_of_int,path)
                    print(top10_info.head(n=5))
        except AttributeError:
            print('Some sort of attribute error, something breaks with relation to PC2, sport:%s' %sports[0])
        
        
#input the first part of the path where you placed the github repository into the second variable.
#remember to include the r before the quotations
perform_trend_analysis('NOC',r'C:\Users\nikhil\Documents\GitHub')
perform_trend_analysis('Height',r'C:\Users\nikhil\Documents\GitHub')
perform_trend_analysis('Weight',r'C:\Users\nikhil\Documents\GitHub')
perform_trend_analysis('Age',r'C:\Users\nikhil\Documents\GitHub')

#perform_PCA_analysis('NOC',r'C:\Users\nikhil\Documents\GitHub')
#perform_PCA_analysis('Height',r'C:\Users\nikhil\Documents\GitHub')
#perform_PCA_analysis('Weight',r'C:\Users\nikhil\Documents\GitHub')
#perform_PCA_analysis('Age',r'C:\Users\nikhil\Documents\GitHub')