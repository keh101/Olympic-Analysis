# -*- coding: utf-8 -*-
"""
Not used to generate any of the data. Just used to store useful functions for future use
"""

df_to = pd.read_csv(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/NOC_Sport_Year.csv")
multiindex = df_to.set_index(['Year','NOC','Sport'])

new_multi = multiindex.drop(['Unnamed: 0'], axis =1)
df2016 = new_multi.loc[2016]
df2016_unstack = df2016.unstack()

os.chdir(r"C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData')
multiindex.loc[2016].to_csv('2016_Country_Sport.csv')

df_to_cheat = pd.read_csv(r"C:\Users\nikhil\Documents\WinPython\notebooks\2016_Country_Sport.csv")
sports_2016 = df_to_cheat.rename(index=str, columns ={'Unnamed: 0':'NOC'})
concatenated_cols = pd.concat([sports_2016,gdp])
minus_country = concatenated_cols.drop(['Country'], axis=1)

firstly = minus_country.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
#drops all columns that contain only NAs
secondly = firstly.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
#replce nan with 0 to ready dead for singular value decomposition/PCA
multi_ind2 = secondly.where((pd.notnull(secondly)),0)


unique_indices= sport_weight.index.unique()
        
just_2016= sport_weight.xs(2016)

arch_weight['Total'] =arch_weight['Weight']*arch_weight['count']

country_ski = ski_multi.loc(axis=0)[:,['%s'%countries[0]]]
if not country_ski.empty:
    countryplot = country_ski.plot()
    countryplot.set_title('%s Ski Jumping'%countries[0])
    
os.chdir(r'C:\Users\nikhil\Desktop\ECE143 HW\Olympic-Analysis-master\Olympic-Analysis-master\120-years-of-olympic-history-athletes-and-results')
with open('sports.csv', 'r') as f:
    reader = csv.reader(f)
    listofsports = list(reader)
with open('noc.csv', 'r') as g:
    reader = csv.reader(g)
    listofnocs = list(reader)
