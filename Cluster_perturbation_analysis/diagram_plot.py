# -*- coding: utf-8 -*-
"""
@author: Iurii Savvateev, Daria Savvateeva
"""

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

import os

from statsmodels.stats.multitest import multipletests


path=r'D:\Parcellation\Github\Cluster_perturbation_analysis\Zmaps_comparison'
condition1='GFP_masked'
condition2='Gi_masked'


c=2 # specify the cluster number 
cortex='pfc' # 'pfc' to display regions assigned to pfc, 'no_pfc' otherwise
significant='ss' #"ss' for displaying only statistical significant, 'no_ss' otherwise
subcortex_status='' #no_subcortex to remove all subcortical areas except Thalamus, for Gozzi data.



def polar_twin(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2.grid(False)
    ax2.set_yticklabels([])
    # There should be a method for this, but there isn't... Pull request?
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()
    # Ensure that original axes tick labels are on top of plots in twinned axes
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)
    return ax2

def data_read(option1,option2,path,cluster_number=1,cortex_status='',subcortex_status='',only_significant='no_ss'):
    """Reads the statistics from two compared options in path
    Inputs: option_1, option_2,data_path, cluster_number,
    Cluster number should be specified. By default the first
    cluster is taken"""


    
    data=pd.read_excel(os.path.join(path,f'comparison-cluster-{cluster_number}-{option1}-{option2}.xlsx'))
    
    
    #Dropping only the lines with both zscores distributions being statistically significantly
    #less than the specified threshold. Threshold is specified in at the head of the Excel column
    
    single_significance=data.columns[data.columns.str.contains('ss', case=False)]
    
    # case argument accounts for lower and upper case values.
    data.drop(data[(data[single_significance[0]]<0.05)&(data[single_significance[1]]<0.05)].index,inplace=True)
    
    # remove data which individually not statistically significant, based on a threshold indicated 
    # For the comparison at least one of the distributions should be statistically higher than the
    # threshold
    
  
    #Always plotting the values
    comparison_count=len(data.index) #used later for Bonferroni correction, since data from 
    
    #Drop data corresponding to PFC,if the corresponding parameter is specified
    
    if cortex_status=='no_pfc':
        data.drop(data[(data['Abbreviation']=='ACAv')| (data['Abbreviation']=='ACAd')\
              |(data['Abbreviation']=='PL')].index,inplace=True) #cluster 5 and 7

        data.drop(data[(data['Abbreviation']=='ORBm')| (data['Abbreviation']=='ORBvl')\
            |(data['Abbreviation']=='ORBl')|(data['Abbreviation']=='ILA')].index,inplace=True)
    if subcortex_status=='no_subcortex':
        data.drop(data[(data['Area']=='Hippocampus')|(data['Area']=='Cortical Subplate')\
                        |(data['Area']=='Striatum')|(data['Area']=='Pallidum')|(data['Area']=='Hypothalamus')].index,inplace=True)

    
    #Checking statistical significance with multiple correction types  
    #Bonferroni correction is chosen as a final one
    
    
    th05=0.05
    th001=0.01
    th0001=0.001
    th00001=0.0001
    
    Cat=data[['Abbreviation', 'p-value']]
    #xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='fdr_tsbky')
    #xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='fdr_bh')
    #xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='fdr_by')
    #xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='fdr_tsbh')
    xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='bonferroni')
    #xx_corr=multipletests(Cat['p-value'].values,alpha=0.05,method='holm-sidak')
    
    Cat['p-value']=xx_corr[1]
    
    
    Cat.loc[(Cat['p-value']>th001) & (Cat['p-value']<th05),['Stars']]='*'
    Cat.loc[(Cat['p-value']>th0001) & (Cat['p-value']<th001),['Stars']]='**'
    Cat.loc[(Cat['p-value']>th00001) & (Cat['p-value']<th0001),['Stars']]='***'
    Cat.loc[(Cat['p-value']<th00001),['Stars']]='****'
    Cat.loc[(Cat['p-value']>th05),['Stars']]=''
    
 
    if only_significant=='ss': #plotting of a subplot with the regions that statistically significantly differ
          
          selected=list(Cat.loc[Cat['p-value']>th05].index) #select indexes with no statistical significance
          data.drop(data[data['Area'].index.isin(selected)].index,inplace=True) #remove corresponding rows
          Cat.drop(Cat[Cat['Stars'].index.isin(selected)].index,inplace=True)
          
    return data, Cat,comparison_count


compared_custer_1=condition1
compared_custer_2=condition2

compared_custer_1_name=condition1[condition1.index("_")+1:] 
compared_custer_2_name=condition2[condition2.index("_")+1:] 


data,Cat_data,bf_count=data_read(condition1,condition2,path,cluster_number=c,\
                                 cortex_status=cortex,subcortex_status=subcortex_status, only_significant=significant)


Cat=Cat_data


area_color={}
area_color['Isocortex']='yellow'
area_color['Hippocampus']='silver'
area_color['Thalamus']='limegreen'
area_color['Striatum']='brown'
area_color['Pallidum']='chocolate'
area_color['Hypothalamus']='pink'
area_color['Cortical Subplate']='aqua'



Cat['Label']=Cat['Abbreviation'] + Cat['Stars']
categories=list(Cat['Label'])
N = len(categories)
 
# The angle of each axis in the plot. Divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  
 

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig.set_size_inches(15, 15)


# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
plt.grid(True)
 
# Draw ylabels
ax.set_rlabel_position(0)

# define max value in the dataset

x6=data[f'{condition1}_mean'].fillna(0).tolist() # this is the data to plot
x7=data[f'{condition2}_mean'].fillna(0).tolist()

r_max=max([max(x6), abs(min(x6)),  max(x7), abs(min(x7))]) 
plt.ylim(0,r_max)

# rotate position of labels to make them readable 
plt.gcf().canvas.draw()
alpha = np.linspace(0, 2*np.pi,len(ax.get_xticklabels())+1)
alpha[np.cos(alpha) < 0] = alpha[np.cos(alpha) < 0] + np.pi
alpha = np.rad2deg(alpha)
labels = []
ally=[]

for label, a in zip(ax.get_xticklabels(), alpha):
    x,y = label.get_position()
    ally+=[y]
    #Labels shift parameter !!!
    lab = ax.text(x,y+0.1, label.get_text(), transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va()) # 0.25 is a shift parameter
    lab.set_rotation(a) # to rotate the labels
    
    if '*' in lab.get_text(): #'*' if the stars notation is used
        lab.set_color('black')
        lab.set_weight('bold')
        lab.set_fontsize(50)
    else:
        lab.set_color('grey')
        lab.set_fontsize(30)
    labels.append(lab)

ax.set_xticklabels([]) # remove old x labels


areas=data.Area.unique()
shift_arg=0.5*angles[1]
area_stats={}

# Specify the variables for the 'donut' filling
ind_old=0
numpoints = 1000

display_arg=2 # To expand or to shrink the shaded area

ax2=polar_twin(ax)
#area='Isocortex'
fill_area_combined={}
#r_fill=round(r_max,0)+4 # for PL-ORB_IL-ORB comparison
# Width of the filling !!!!!!
r_fill=round(r_max,0)+5 # for PL-ORB_IL-ORB comparison
#ax.plot([-0.5*angles[1],-0.5*angles[1]], [0,r_max], linewidth=2, color='k' ) # first line
data_new_index=data.reset_index()
for count, area in enumerate(areas):
        ind=data_new_index[data_new_index.Area==area].index.max()
        
        if count == len(areas)-1:  # condition for the last filling, count starts at zero
            fill_area=np.linspace(ind_old*angles[1]+shift_arg, ind*angles[1]+shift_arg, numpoints)
        elif count == 0: # condition for the first filling
            fill_area=np.linspace(ind_old*angles[1]-shift_arg, ind*angles[1]+shift_arg, numpoints)
        else:
            fill_area=np.linspace(ind_old*angles[1]+shift_arg, ind*angles[1]+shift_arg, numpoints)
            
        area_stats[area]=data[data.Area==area].index.value_counts().sum() # Dictionary contatining statistics not displayed
        fill_area_combined[area]=fill_area
            
        # previously used pie
        ax2.fill_between(fill_area, [r_fill+display_arg]*numpoints, [r_fill-display_arg]*numpoints, color=area_color[area], alpha=0.25)
       
    
        ind_old=ind

# generating non-displayed data to use the linked labels        
for regions,regions_data in fill_area_combined.items():
    ax2.plot(regions_data, [r_fill]*numpoints, color=area_color[regions], alpha=0.1,label=regions)


  
# # Ind1
x6 +=x6[:1]
x7 +=x7[:1]

ax.plot(angles, [abs(x) for x in x6], linewidth=5, linestyle='solid',color='b', label=f"{compared_custer_1_name}")
y1_err=[abs(x)-data.iloc[ind_x][f'{condition1}_std'] if ind_x <len(data) else \
                       abs(x)-data.iloc[0][f'{condition1}_std'] for ind_x,x in enumerate(x6)]
y2_err=[abs(x)+data.iloc[ind_x][f'{condition1}_std'] if ind_x <len(data) else \
                       abs(x)+data.iloc[0][f'{condition1}_std'] for ind_x,x in enumerate(x6)]
    
ax.fill_between(angles,y1_err,y2_err,facecolor='b',alpha=0.3)


ax.plot(angles, [abs(x) for x in x7], linewidth=5, linestyle='solid', color='r', label=f"{compared_custer_2_name}")
y1_err=[abs(x)-data.iloc[ind_x][f'{condition2}_std'] if ind_x <len(data) else \
                       abs(x)-data.iloc[0][f'{condition2}_std'] for ind_x,x in enumerate(x7)]
y2_err=[abs(x)+data.iloc[ind_x][f'{condition2}_std'] if ind_x <len(data) else \
                       abs(x)+data.iloc[0][f'{condition2}_std'] for ind_x,x in enumerate(x7)]
    
ax.fill_between(angles,y1_err,y2_err,facecolor='r',alpha=0.3)
    

ax.spines['polar'].set_visible(False) # Remove the outside border


ax.set_ylim([-2, r_fill+display_arg]) 
ax2.set_ylim([-2, r_fill+display_arg])

y_labels=[n.get_text() for n in ax.get_yticklabels()] # only display the first values to avoid the overlap between the labels and the radii.
ax.set_yticklabels(y_labels[:2])

# Show the graph
save_path=os.path.join(path,'Plots')
plt.savefig(f'{save_path}\Cluster_{c}_{cortex}_{significant}_stim.png', transparent=True, bbox_inches="tight", dpi=300)
#plt.show()
