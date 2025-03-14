# -*- coding: utf-8 -*-
"""

@author: Daria Savvateeva, Iurii Savvateev
"""

import matplotlib.pyplot as plt
import numpy as np

# Define Data 



##################### 4 cluster solution, PFC-ALL ################################## 

####### 4 cluster solution for kmeans pfc-all with 20% data
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=4
# method='kmeans'
# data_size=20

# Data1 = np.array([0,       92.5,  3.6,     0,   93.7,    58.4,  25])
# Data2 = np.array([81.1,    0,     52.7,    0,   0,       0,     0])
# Data3 = np.array([18.5,    0,     42.4,    0,   0,       0,     64.6])
# Data4 = np.array([0.4,     7.5,   1.3,   100,   6.3,     41.6,  10.4])


####### 4 cluster solution for kmeans pfc-all with 40% data
# Data1 = np.array([0,      100,    3.1,     0,    91.1,    53.5,   16.7])
# Data2 = np.array([79.2,    0,     59.8,    0,       0,    0,       1.4])
# Data3 = np.array([20.8,    0,     38.8,    0,       0,    0,        75])
# Data4 = np.array([0,       0,     0,       100,   8.9,    46.5,    6.9])
# cluster_title=4
# method='kmeans'
# data_size=40

####### 4 cluster solution for kmeans pfc-all with 60% data

# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=4
# method='kmeans'
# data_size=60

# Data1 = np.array([0,       100,     4.9,     0,    92.4,    55.4,    27.1])
# Data2 = np.array([80.8,    0,       50.9,    0,       0,      0,       0])
# Data3 = np.array([19.2,    0,       44.2,    0,       0,      0,    66])
# Data4 = np.array([0,       0,       0,       100,   7.6,    44.6,       6.9])


# #### 4 cluster solution for kmeans pfc-all with 80% data
# Data1 = np.array([0,    100,     3.6,        0,   91.1,   55.4,    18.1])
# Data2 = np.array([80.8,    0,     53.1,      0,    0,        0,    0])
# Data3 = np.array([19.2,    0,     43.3,      0,    0,        0,    77.8])
# Data4 = np.array([0,       0,     0,       100,    8.9,   44.6,    4.2])
# cluster_title=4
# method='kmeans'
# data_size=80

#### 4 cluster solution for kmeans pfc-all with 100% data
# Data1 = np.array([0,    100,     4.9,       0,   92.4,   55.4,    28.5])
# Data2 = np.array([80,    0,     51.3,       0,      0,      0,       0])
# Data3 = np.array([20,    0,     43.8,       0,      0,      0,    65.3])
# Data4 = np.array([0,     0,        0,     100,    7.6,   44.6,     6.2])

# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=4
# method='kmeans'
# data_size=100


# #### 4 cluster solution for gmmm pfc-all with 100% data
# Data1 = np.array([0,    100,     5.4,       0,   92.4,   55.4,    29.2])
# Data2 = np.array([79.2,    0,     49.6,       0,      0,      0,       0])
# Data3 = np.array([20.8,    0,     45.1,       0,      0,      0,    62.5])
# Data4 = np.array([0,     0,        0,     100,    7.6,   44.6,     8.3])

# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=4
# method='gmm'
# data_size=100


# #### 4 cluster solution for Hierarchical pfc-all with 100% data
# Data1 = np.array([0,      70.1,      0.4,       1.9,   92.4,   72.2,    62.4])
# Data2 = np.array([91.3,      0,     67.4,       0,      0,      0,       0])
# Data3 = np.array([8.7,    28.4,     32.1,       1.9,      0,   26.6,    13.9])
# Data4 = np.array([0,       1.5,        0,     96.3,    7.6,     1.3,    23.8])

# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=4
# method='Hierarchical'
# data_size=100

##################### 7 cluster solution, PFC-PFC ################################## 


# ### 7 cluster solution for kmeans pfc-pfc with 20% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='kmeans'
# data_size=20

# Data1 = np.array([0.4,    1.5,   0.9,     100,    3.8,     34.7,   0.7])
# Data2 = np.array([46,     0,    32.6,     0,      0,       0,      7.6])
# Data3 = np.array([39.2,   0,    43.3,     0,      0,       0,      0])
# Data4 = np.array([0,      58.2,  0.9,     0,      50.6,    29.7,   4.9])
# Data5 = np.array([0,      40.3,  0.9,     0,      40.5,    35.6,   2.1])
# Data6 = np.array([9.8,    0,    10.3,     0,      2.5,     0,      40.3])
# Data7 = np.array([4.5,    0,    11.2,     0,      2.5,     0,      44.4])



# ### 7 cluster solution for kmeans pfc-pfc with 40% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='kmeans'
# data_size=40

# Data1 = np.array([1.1,    0,      0,        100,    5.1,     33.7,   0.7])
# Data2 = np.array([46.4,   0,      27.7,     0,        0,     0,      4.2])
# Data3 = np.array([41.1,   0,      46.9,     0,        0,     0,      0])
# Data4 = np.array([0,      59.7,   0.9,      0,     50.6,     30.7,   4.9])
# Data5 = np.array([0,      40.3,   0.9,      0,     40.5,     35.6,   2.1])
# Data6 = np.array([6.8,    0,      11.2,     0,      1.3,     0,      41.7])
# Data7 = np.array([4.5,    0,      12.5,     0,      2.5,     0,      46.5])

# ### 7 cluster solution for kmeans pfc-pfc with 60% data 
Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
cluster_title=7
method='kmeans'
data_size=60

Data1 = np.array([0.4,     3,       0,       100,    3.8,    37.6,   0])
Data2 = np.array([42.3,    0,       35.7,      0,      0,    0,      17.4])
Data3 = np.array([54.3,    0,       50,        0,      0,    0,      0])
Data4 = np.array([0,       44.8,    0,         0,   46.8,    24.8,   0.7])
Data5 = np.array([0,       32.8,    0,         0,     38,    34.7,   0.7])
Data6 = np.array([2.6,     7.5,     7.1,       0,    5.1,    2,      37.5])
Data7 = np.array([0.4,     11.9,    7.1,       0,    6.3,    1,      43.8])


# ### 7 cluster solution for kmeans pfc-pfc with 80% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='kmeans'
# data_size=80

# Data1 = np.array([1.1,    0,      0,        100,    3.8,     30.7,     1.4])
# Data2 = np.array([46.4,   0,      34.4,     0,        0,     0,        6.2])
# Data3 = np.array([44.5,   0,      44.2,     0,        0,     0,          0])
# Data4 = np.array([0,      40.3,   0.9,      0,     39.2,     36.6,     4.2])
# Data5 = np.array([0,      52.2,   0.4,      0,     50.6,     26.7,     3.5])
# Data6 = np.array([3.4,    7.5,    5.4,      0,      6.3,     5.9,     31.2])
# Data7 = np.array([4.5,    0,      14.7,     0,        0,     0,       53.5])


# ### 7 cluster solution for kmeans pfc-pfc with 100% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='kmeans'
# data_size=100

# Data1 = np.array([0.4,    0,         0,    100,      2.5,    29.7,     0])
# Data2 = np.array([43.4,   0,      36.2,      0,        0,       0,  14.6])
# Data3 = np.array([51.3,   0,      48.7,      0,        0,       0,     0])
# Data4 = np.array([0,      44.8,      0,      0,     48.1,     32.7,    0])
# Data5 = np.array([0,      40.3,    0.9,      0,     40.5,     35.6,  1.4])
# Data6 = np.array([4.2,    0,       7.6,      0,      2.5,     0,    35.4])
# Data7 = np.array([0.8,    14.9,    6.7,      0,      6.3,     2,    48.6])

# ### 7 cluster solution for gmm pfc-pfc with 100% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='gmm'
# data_size=100

# Data1 = np.array([0.4,    4.5,       0,    100,      2.5,    36.6,       0])
# Data2 = np.array([38.5,   0,      34.4,      0,        0,       0,    19.4])
# Data3 = np.array([58.1,   0,      51.8,      0,        0,       0,       0])
# Data4 = np.array([0,      49.3,    0.4,      0,     51.9,     19.8,    4.9])
# Data5 = np.array([0,      26.9,      0,      0,     30.4,     35.6,      0])
# Data6 = np.array([2.6,    9,       8.5,      0,      6.3,        1,    42.4])
# Data7 = np.array([0.4,    10.4,    4.9,      0,      8.9,      6.9,    33.3])

# ### 7 cluster solution for Hierarchical pfc-pfc with 100% data 
# Class = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
# cluster_title=7
# method='Hierarchical'
# data_size=100

# Data1 = np.array([0,         0,         0,    97.2,      2.5,    42.6,       0.7])
# Data2 = np.array([18.9,      0,         8,       0,        0,       0,         0])
# Data3 = np.array([45.7,      0,      48.7,       0,        0,       0,         0])
# Data4 = np.array([0,      65.7,       0.9,       0,     59.5,    25.7,       7.6])
# Data5 = np.array([0,      29.9,       0.4,       0,     32.9,    28.7,         0])
# Data6 = np.array([32.5,      0,      31.7,       0,        0,       0,      22.2])
# Data7 = np.array([3,       4.5,      10.3,     2.8,      5.1,       3,      69.4])



# Specifying colors for 4 cluster solution

# Data=[Data1, Data2, Data3, Data4]
# Col=['orange','deepskyblue','violet','red']# for 4 cluster solution
# Label=['1','2','3','4']
# labels_new=['C3','C1','C2','C4'] # for 4 cluster solution


#Specifying colors for 7 cluster solution

Data=[Data1, Data2, Data3, Data4, Data5, Data6, Data7]

red = np.array([1, 0, 0])
violet = np.array([0.8, 0, 0.9])
blue=np.array([0, 191/255, 1])
dark_blue=np.array([63/255, 72/255, 204/255])
pink=np.array([239/255, 136/255, 190/255])
yellow=np.array([212/255, 175/255, 55/255])
orange=np.array([1, 0.6, 0.1, 1])

Col=[red,blue ,dark_blue ,orange, yellow,violet,pink ]
Label=['1','2','3','4','5','6','7'] 
labels_new=['C4','C1(ventral)','C1(dorsal)','C3(right)','C3(left)','C2(left)','C2(right)'] # for 7 cluster solution


# Define width of stacked chart
w = 0.6

# Plot stacked bar chart
fig=plt.figure(figsize=(20,20))
ax = fig.add_subplot(1, 1, 1)
for ibar in range(len(Data)):
    plt.bar(Class, Data[ibar], w, bottom=sum(Data[:ibar]), color=Col[ibar], label=Label[ibar])

plt.xlabel('Brain region', fontsize=55,fontweight="bold")
plt.ylabel('Percentage', fontsize=55, fontweight="bold")
plt.text(-1,105,f'{cluster_title} cluster solution for {method} on {data_size} mice', fontsize=55,fontweight="bold")
#plt.rcParams['axes.titley'] = 
plt.ylim(0,100)
plt.xticks(rotation='horizontal', fontsize=45)
plt.yticks(rotation='horizontal', fontsize=45)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=(1.04, 0), fontsize=45,ncol =len(Label))

#ax.legend(handles[::-1], labels_new[::-1], loc=(1.04, 0), fontsize=45,ncol =len(labels_new))
ax.legend(handles[::-1], labels_new[::-1], loc=(1.04, 0), fontsize=45)

for bar in ax.patches:
    height = bar.get_height()
    if height>3:
        width = bar.get_width()
        x = bar.get_x()
        y = bar.get_y()
        label_text = round(height,1)
        label_x = x + width / 2
        label_y = y + height / 2
        ax.text(label_x, label_y, label_text, ha='center',    
                va='center', fontsize=40,fontweight="bold")
    
plt.show()