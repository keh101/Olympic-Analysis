# -*- coding: utf-8 -*-
"""
Split files organized by year into individual csv files
"""
import csv
import os
from itertools import groupby


#drop columns manually
wrestles_df = multi_ind.loc['Wrestling']
non_null_columns = [col for col in wrestles_df.columns if wrestles_df.loc[:, col].notna() >10]
wrestles_df[non_null_columns]

#Age vs sport vs year
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/AgevSport")
for key, rows in groupby(csv.reader(open("Age_Sport_Year.csv")),
                         lambda row: row[1]):
    with open("%s.csv" % key, "w") as output:  
        header_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_writer.writerow(['lindex','Sport','Year','Age','count'])
        for row in rows:
            output.write(','.join(row) + '\n')
            
#Height vs sport vs year
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/HeightvSport")
for key, rows in groupby(csv.reader(open("Height_Sport_Year.csv")),
                         lambda row: row[1]):
    with open("%s.csv" % key, "w") as output:
        header_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_writer.writerow(['lindex','Sport','Year','Height','count'])
        for row in rows:
            output.write(",".join(row) + "\n")
            
#Country vs sport vs year
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/CountryvSport")
for key, rows in groupby(csv.reader(open("Country_Sport_Year.csv")),
                         lambda row: row[1]):
    with open("%s.csv" % key, "w") as output:
        header_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_writer.writerow(['lindex','Sport','Year','Country','count'])
        for row in rows:
            output.write(",".join(row) + "\n")
            
#Weight vs sport vs year
os.chdir("C:/Users/nikhil/Desktop/ECE143 HW/Olympic-Analysis-master/AnalyzedData/WeightvSport")
for key, rows in groupby(csv.reader(open("Weight_Sport_Year.csv")),
                         lambda row: row[1]):     
    with open("%s.csv" % key, "w") as output:
        header_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header_writer.writerow(['lindex','Sport','Year','Weight','count'])
        for row in rows:
            output.write(",".join(row) + "\n")