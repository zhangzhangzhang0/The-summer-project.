


import scipy.stats as stats
import csv
csv_file = open("Mybalances3.csv","r")
csv_reader = csv.reader(csv_file);

y1 = []
y2 = []

name1 = None
name2 = None



count = 0
for item in csv_reader:
    y2.append(int(float(item[11])))  # Xgboost
    y1.append(int(float(item[7])))  #GVWY
    # y1.append(int(float(item[11])))  # GVWY
    # y1.append(int(float(item[23])))  # ZIC
    # y1.append(int(float(item[27])))  # ZIP
    #y1.append(int(float(item[15])))  # SHVR

    name2 = item[8] #Xgboost
    name1 = item[4]
    # name1 = item[8]  # GVWY
    # name1 = item[12] # SHVR
    # name1 = item[20] #ZIC
    # name1 = item[24]   # ZIP
    count += 1


u_statistic, pVal = stats.mannwhitneyu(y1, y2,alternative='less')

print ("u_statistic is %f"%u_statistic)
print ("p value is %f"%pVal)


import numpy as np
#create 95% confidence interval for population mean weight
print (np.mean(y2))
print (stats.t.interval(confidence=0.95, df=len(y2)-1, loc=np.mean(y2), scale=stats.sem(y2))[1] - np.mean(y2))
print ("")
print (np.mean(y1))
print (stats.t.interval(confidence=0.95, df=len(y1)-1, loc=np.mean(y1), scale=stats.sem(y1))[1] - np.mean(y1))
print ("")

df = []

for index in range(len(y1)):
    df.append(y2[index]-y1[index])
print (np.mean(df))
print (stats.t.interval(confidence=0.95, df=len(df)-1, loc=np.mean(df), scale=stats.sem(df))[1] - np.mean(df))
