

from matplotlib import pyplot as plot
import csv
import random






csv_file = open("Mybalances3.csv","r")
csv_reader = csv.reader(csv_file);

y1 = []
y2 = []

name1 = None
name2 = None


cy1 = 0;
cy2 = 0;
count = 0
for item in csv_reader:

    #y2.append(int(float(item[19])))  # Xgboost
    y1.append(int(float(item[7])))  #GVWY
    y2.append(int(float(item[11])))  # Xgboost
    #y1.append(int(float(item[23])))  # ZIC
    #y1.append(int(float(item[27])))  # ZIP
    #y1.append(int(float(item[15])))  # SHVR



    cy1 += int(float(item[7]))
    #cy1 += int(float(item[11]))
    #cy1 += int(float(item[23]))
    #cy1 += int(float(item[27]))
    # cy1 += int(float(item[15]))
    cy2 += int(float(item[11]))

    name2 = item[8]
    name1 = item[4]  # AA
    # name1 = item[8]  # GVWY
    # name1 = item[12] # SHVR
    # name1 = item[20] #ZIC
    # name1 = item[24]   # ZIP
    count += 1
x = range(1,count+1)
fig, ax = plot.subplots()

cost1 = cy1/count
cost1_list = [cost1 for i in range(1,count+1)]

cost2 = cy2/count
cost2_list = [cost2 for i in range(1,count+1)]

ax.plot(x,y1,label="AA")
ax.plot(x,y2,label="Xgboost")
ax.plot(x,cost1_list,linestyle="dashed", label= "AA's average profit = "+str(float((cy1+0.0)/count)))
ax.plot(x,cost2_list,linestyle="dashed", label= "Xgboost's average profit = "+str(float((cy2+0.0)/count)))
# ax.plot(x,y3,label=name3)
# ax.plot(x,y4,label=name4)

# xticks_label  =  [i*5 for i in range(1,21)]
# plot.xticks( xticks_label)
# yticks_label  =  [i*5 for i in range(10,30)]
# plot.yticks( yticks_label)
ax.set_xlabel('trading day')
ax.set_ylabel('total profit in each trading day')
ax.legend()
plot.savefig("balance_AA.png")
plot.show()
