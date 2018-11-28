# -*- coding: utf-8 -*-
import pandas as pd
import os


"""
Here we partitioned the data into relevant confounding variables we are hoping
to analyze through chi square analysis
"""
df = pd.read_excel("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/Olympic-Analysis-master/Statiscal Analysis/multiple variables.xlsx")
m=df
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData")


#get height vs. sport counts
hieghtvsport = m.groupby(['Height','Sport']).size().to_frame('count').reset_index()
hieghtvsport.to_csv('heightvsport.csv')
#turn column format to table format
restructheight = hieghtvsport.pivot(index = 'Sport',columns = 'Height', values='count')
restructheight.to_csv('Height_struct.csv')
#get year vs. height vs. sport counts
hieghtvsportvyear = m.groupby(['Year','Sport','Height']).size().to_frame('count').reset_index()
hieghtvsportvyear.to_csv('height_sport_year.csv')
#split height_sport_year.csv into tables of counts for each year
df2 = hieghtvsportvyear.set_index('Year').T.reset_index(drop=True) \
    .T.groupby(level=0).apply(lambda df: df.reset_index(drop=True)) \
    .stack().unstack(1).T

df2.columns = df2.columns.set_levels((df2.columns.levels[1] + 1).astype(str), level=1)
df2.columns = df2.columns.to_series()

df2
df2.to_csv('split_height.csv')

#get weight vs. sport counts
weightvsport = m.groupby(['Weight','Sport']).size().to_frame('count').reset_index()
weightvsport.to_csv('Weight_Sport.csv')
#turn column format to table format
restructweight = weightvsport.pivot(index = 'Sport',columns = 'Weight', values='count')
restructweight.to_csv('Weight_struct.csv')
#get year vs. weight vs. sport counts
yearvweightvsport = m.groupby(['Year','Sport','Weight']).size().to_frame('count').reset_index()
yearvweightvsport.to_csv('Weight_Sport_Year.csv')
#split Weight_Sport_Year.csv into tables of counts for each year
df2 = yearvweightvsport.set_index('Year').T.reset_index(drop=True) \
    .T.groupby(level=0).apply(lambda df: df.reset_index(drop=True)) \
    .stack().unstack(1).T

df2.columns = df2.columns.set_levels((df2.columns.levels[1] + 1).astype(str), level=1)
df2.columns = df2.columns.to_series()

df2
df2.to_csv('split_weight.csv')

#get age vs. sport counts
agevsport = m.groupby(['Age','Sport']).size().to_frame('count').reset_index()
agevsport.to_csv('Age_Sport.csv')
#turn column format to table format
restructage = agevsport.pivot(index = 'Sport',columns = 'Age', values='count')
restructage.to_csv('Age_struct.csv')
#get year vs. age vs. sport counts
yearvagevsport = m.groupby(['Year','Sport','Age']).size().to_frame('count').reset_index()
yearvagevsport.to_csv('Age_Sport_Year.csv')
#split Age_Sport_Year.csv into tables of counts for each year
df2 = yearvagevsport.set_index('Year').T.reset_index(drop=True) \
    .T.groupby(level=0).apply(lambda df: df.reset_index(drop=True)) \
    .stack().unstack(1).T

df2.columns = df2.columns.set_levels((df2.columns.levels[1] + 1).astype(str), level=1)
df2.columns = df2.columns.to_series()

df2
df2.to_csv('split_age.csv')


#get country vs. sport counts
countryvsport = m.groupby(['NOC','Sport']).size().to_frame('count').reset_index()
countryvsport.to_csv('Country_Sport.csv')
#turn column format to table format
restructcountry = countryvsport.pivot(index = 'Sport',columns = 'NOC', values='count')
restructcountry.to_csv('Country_struct.csv')
#get country vs. sport counts
countryvsportvyear = m.groupby(['Year','Sport','NOC']).size().to_frame('count').reset_index()
countryvsportvyear.to_csv('Country_Sport_Year.csv')
#split Country_Sport_Year.csv into tables of counts for each year
df2 = countryvsportvyear.set_index('Year').T.reset_index(drop=True) \
    .T.groupby(level=0).apply(lambda df: df.reset_index(drop=True)) \
    .stack().unstack(1).T

df2.columns = df2.columns.set_levels((df2.columns.levels[1] + 1).astype(str), level=1)
df2.columns = df2.columns.to_series()

df2
df2.to_csv('split_country.csv')


