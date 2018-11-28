# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:04:53 2018

@author: nikhil
"""

#read in raw data
m = pd.read_excel("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/Olympic-Analysis-master/Statiscal Analysis/multiple variables.xlsx")


#get year vs. height vs. sport counts from input excel file
def get_height_sport_year(m, df_to=0):
    if df_to ==0:
        os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")
        hieghtvsportvyear = m.groupby(['Sport','Year','Height']).size().to_frame('count').reset_index()
        hieghtvsportvyear.to_csv('Height_Sport_Year.csv')
        df = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/Height_Sport_Year.csv")
    else:
        df =df_to
    multiindex_height = df.set_index(['Sport','Year','Height'])
    return multiindex_height
#get year vs. weight vs. sport counts
def get_weight_sport_year(m, df_to=0):
    if df_to ==0:
        os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")
        yearvweightvsport = m.groupby(['Sport','Year','Weight']).size().to_frame('count').reset_index()
        yearvweightvsport.to_csv('Weight_Sport_Year.csv')
        df = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/Weight_Sport_Year.csv")
    else:
        df =df_to
    multiindex_weight = df.set_index(['Sport','Year','Weight'])
    return multiindex_weight
#get year vs. age vs. sport counts from input excel file
def get_age_sport_year(m, df_to=0):
    if df_to ==0:
        os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")
        yearvagevsport = m.groupby(['Sport','Year','Age']).size().to_frame('count').reset_index()
        yearvagevsport.to_csv('Age_Sport_Year.csv')
        df = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/Age_Sport_Year.csv")
    else:
        df =df_to
    multiindex_age = df.set_index(['Sport','Year','Age'])
    return multiindex_age
#get year vs. country vs. sport counts from input excel file
def get_country_sport_year(m, df_to=0):
    if df_to ==0:
        os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")
        countryvsportvyear = m.groupby(['Sport','Year','NOC']).size().to_frame('count').reset_index()
        countryvsportvyear.to_csv('Country_Sport_Year.csv')
        df = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/Country_Sport_Year.csv")
    else:
        df =df_to
    multiindex_country = df.set_index(['Sport','Year','Country'])
    return multiindex_country