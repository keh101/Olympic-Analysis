# -*- coding: utf-8 -*-

"""
Graphing averages of interesting data
"""
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
 



def graph_NOC_counts(path):
    with open(r'%s\Olympic-Analysis-master\Olympic-Analysis-master\120-years-of-olympic-history-athletes-and-results\noc.csv'%path, 'r') as g:
        reader = csv.reader(g)
        listofnocs = list(reader)
    df_to = pd.read_csv(r"%s/Olympic-Analysis-master/AnalyzedData/NOC_Sport_Year.csv"%path)
    multiindex = df_to.set_index(['Year','NOC','Sport'])
    new_multi = multiindex.drop(['Unnamed: 0'], axis =1)
    sport_noc_list = ['Ski Jumping', 'Athletics', 'Biathlon']
    for sports in sport_noc_list:    
        for countries in listofnocs:
            ski_multi = new_multi.xs('%s'%sports, level=2)
            country_ski = ski_multi.loc(axis=0)[:,['%s'%countries[0]]]
            if not country_ski.empty:
                countryplot = country_ski.plot()
                countryplot.set_title('%s Ski Jumping'%countries[0])
            
def graph_weight_counts(path):
    df_to = pd.read_csv(r"%s/Olympic-Analysis-master/AnalyzedData/Weight_Sport_Year.csv"%path)
    multiindex = df_to.set_index(['Year','Weight','Sport'])
    new_multi = multiindex.drop(['Unnamed: 0'], axis =1)
    weight_sport_list = ['Archery','Athletics','Biathlon','Beach Volleyball','Canoeing','Cycling','Snowboarding','Softball','Sailing']    
    
    for sports in weight_sport_list:
        sport_multi = new_multi.xs('%s'%sports, level=2)
        sport_weight= sport_multi.reset_index(level='Weight')
        sport_weight['Total']=sport_weight['Weight']*sport_weight['count']
        
        totals_sum = sport_weight.groupby(level=0)['Total'].sum()
        counts_sum = sport_weight.groupby(level=0)['count'].sum()
        averages = totals_sum/counts_sum
        plt.figure()
        average_plot = averages.plot()
        average_plot.set_title('%s Average Weight Over Time'%sports)
        average_plot.set_ylabel('Weight')
        average_plot.get_figure().savefig(r'%s\Olympic-Analysis-master\AnalyzedData\Weight\Average\%s_Average.png' %(path,sports))
        
            
def graph_age_counts(path):
    df_to = pd.read_csv(r"%s/Olympic-Analysis-master/AnalyzedData/Age_Sport_Year.csv"%path)
    multiindex = df_to.set_index(['Year','Age','Sport'])
    new_multi = multiindex.drop(['Unnamed: 0'], axis =1)
    age_sport_list = ['Art Competitions','Athletics','Basketball', 'Biathlon', 
                         'Cross Country Skiing', 'Football', 'Golf', 'Gymnastics', 
                         'Hockey', 'Judo', 'Rowing', 'Sailing', 'Snowboarding','Softball', 
                         'Swimming', 'Water Polo', 'Wrestling']    
    for sports in age_sport_list:
        sport_multi = new_multi.xs('%s'%sports, level=2)
        sport_age= sport_multi.reset_index(level='Age')
        sport_age['Total']=sport_age['Age']*sport_age['count']
        
        totals_sum = sport_age.groupby(level=0)['Total'].sum()
        counts_sum = sport_age.groupby(level=0)['count'].sum()
        averages = totals_sum/counts_sum
        plt.figure()
        average_plot = averages.plot()
        average_plot.set_title('%s Average Age Over Time'%sports)
        average_plot.set_ylabel('Age')
        average_plot.get_figure().savefig(r'%s\Olympic-Analysis-master\AnalyzedData\Age\Average\%s_Average.png' %(path,sports))

#input path to github repository
graph_NOC_counts(r'C:\Users\nikhil\Desktop\ECE143 HW')       
graph_age_counts(r'C:\Users\nikhil\Desktop\ECE143 HW')
graph_weight_counts(r'C:\Users\nikhil\Desktop\ECE143 HW')  

