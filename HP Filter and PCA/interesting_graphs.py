# -*- coding: utf-8 -*-

"""
Graphing averages of interesting data
"""
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
 



def graph_NOC_counts():
    with open(r'.\120-years-of-olympic-history-athletes-and-results\noc.csv', 'r') as g:
        reader = csv.reader(g)
        listofnocs = list(reader)
    df_to = pd.read_csv(r"./AnalyzedData/NOC_Sport_Year.csv")
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
            
def graph_weight_counts():
    df_to = pd.read_csv(r"./AnalyzedData/Weight_Sport_Year.csv")
    multiindex = df_to.set_index(['Year','Weight','Sport'])
    new_multi = multiindex.drop(['Unnamed: 0'], axis =1)
    weight_sport_list = ['Archery','Athletics','Biathlon','Beach Volleyball','Canoeing','Cycling','Snowboarding','Softball','Sailing']    
    
    for sports in weight_sport_list:
        print(sports)
        sport_multi = new_multi.xs('%s'%sports, level=2)
        print(sport_multi.head(n=5))
        print(1)
        sport_weight= sport_multi.reset_index(level='Weight')
        sport_weight['Total']=sport_weight['Weight']*sport_weight['count']
        print(sport_weight.head(n=5))
        print(1)
        totals_sum = sport_weight.groupby(level=0)['Total'].sum()
        print(totals_sum.head(n=5))
        print(1)
        counts_sum = sport_weight.groupby(level=0)['count'].sum()
        print(counts_sum.head(n=5))
        print(1)
        averages = totals_sum/counts_sum
        print(averages)
        print(1)
        plt.figure()
        average_plot = averages.plot()
        average_plot.set_title('%s Average Weight Over Time'%sports)
        average_plot.set_ylabel('Weight')
        average_plot.get_figure().savefig(r'.\AnalyzedData\Weight\Average\%s_Average.png' %sports)
        
            
def graph_age_counts():
    df_to = pd.read_csv(r"./AnalyzedData/Age_Sport_Year.csv")
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
        average_plot.get_figure().savefig(r'.\AnalyzedData\Age\Average\%s_Average.png' %sports)

graph_NOC_counts()       
graph_age_counts()
graph_weight_counts()  

