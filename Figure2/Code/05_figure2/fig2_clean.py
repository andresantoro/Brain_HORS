#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:27:42 2024

@author: andrea santoro
"""

import numpy as np
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm as tqdm_notebook
import glob
#import scipy.io as sio
#import h5py
import networkx as nx
#from itertools import combinations
#from scipy.stats import pearsonr,zscore
import pandas as pd
#import copy
#from scipy.io import loadmat
#from scipy.spatial.distance import cosine
#import pandas as pd
#import matplotlib
#from multiprocessing import Pool,Process

import math 

###Clustering
#import networkx as nx
from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim
#from sklearn.feature_selection import mutual_info_regression
# from minepy import cstats,MINE
# import sys
# sys.path.append('./')
# from utils_plotting import plot_adjustments

## Plotting 
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec
from palettable.colorbrewer.qualitative import Dark2_7
from palettable.colorbrewer.sequential import *
from matplotlib.colors import ListedColormap
import brewer2mpl



bmap = brewer2mpl.get_map('Accent', 'qualitative', 3)
bmap_used=bmap.hex_colors
bmap_used[0]='#386cb0'


## Plot boxes for the HCP cognitive tasks
def plot_boxes(color='red', lw=2, linestyles='solid', ax=None):
    #REST,EMO,GAM,LAN,MOT,REL,SOC,WM
    ##300,144,156,273,170,138,128,312
    box_cumsum = np.cumsum([0, 300, 144, 156, 273, 170, 138, 128, 312])
    # micro_tasks=[0,300,#REST
    #              66,78,#EMOTION
    #              78,78,#GAMBLING
    #              139,134,#LANGUAGE
    #              34,34,34,34,34,#MOTOR
    #              69,69,#RELATIONAL
    #              64,64,#SOCIAL
    #              39,39,39,39,39,39,39,39,#WM
    #              ]
    palette_tab10 = sns.color_palette("tab10", 10)
    for i in range(len(box_cumsum)-1):
        x_p, x_f = box_cumsum[i]-1, box_cumsum[i+1]-1
        if i != 7:
            color = palette_tab10[i]
        else:
            color = palette_tab10[i+1]
        if ax == None:
            plt.vlines(x_p, x_p, x_f, color=color,
                       lw=lw, linestyles=linestyles)
            plt.vlines(x_f, x_p, x_f, color=color,
                       lw=lw, linestyles=linestyles)

            plt.hlines(x_p, x_p, x_f, color=color,
                       lw=lw, linestyles=linestyles)
            plt.hlines(x_f, x_p, x_f, color=color,
                       lw=lw, linestyles=linestyles)
        else:
            ax.vlines(x_p, x_p, x_f, color=color, lw=lw, linestyles=linestyles)
            ax.vlines(x_f, x_p, x_f, color=color, lw=lw, linestyles=linestyles)

            ax.hlines(x_p, x_p, x_f, color=color, lw=lw, linestyles=linestyles)
            ax.hlines(x_f, x_p, x_f, color=color, lw=lw, linestyles=linestyles)
            
def select_task_file(filename,map_label_intervals,task='EMO'):
    data=np.array(sorted(np.loadtxt(filename),key=lambda x:x[0]))
    idx1,idx2=map_label_intervals[task][0],map_label_intervals[task][1]
    return(data[idx1:idx2,:])
    
def all_task_file(filename,map_label_intervals):
    data=np.array(sorted(np.loadtxt(filename),key=lambda x:x[0]))
    data_all={i:[] for i in map_label_intervals}
    for task in map_label_intervals:
        idx1,idx2=map_label_intervals[task][0],map_label_intervals[task][1]
        data_all[task]=data[idx1:idx2,:]
    return(data_all)

# def path_data_task(pathdata,subjectID,taskID,LR_flag):
#     if taskID == 'REST1':
#         c_path=f'{pathdata}HOindicators_{subjID}_rest1_{LR_flag}'
#     else:
#         c_path=f'{pathdata}HO_indicators_{subjID}_{taskID}_{LR_flag}.txt'
#     return  c_path

def load_data(filename):
    data=np.array(sorted(np.loadtxt(filename),key=lambda x:x[0]))
    print("loaded: ",filename)
    return(data)

### Function that maps a 3-tuple into the 2-dimensional coordinate within a triangle  
def triple2ternary(triple, pw=1):
    # normalize 
    triple = [float(i)/sum(triple) for i in triple]
    y = triple[1] * math.sin(math.radians(60));
    x = 1 - triple[0] - y * 1/math.tan(math.radians(60));
    y = 1+y
    return pw*x,pw*y

def extract_triplets_brain(braindata,plotwidth=300):
    chaos_triplets=braindata[0][:,2:5]
    for i in range(1,len(braindata)):
        current_behaviour=braindata[i][:,2:5]
        chaos_triplets=np.concatenate((chaos_triplets,current_behaviour))
    list_synthetic=[]
    for i in chaos_triplets:
        x,y=triple2ternary(i, plotwidth)
        list_synthetic.append([x,y])
    list_synthetic=np.array(list_synthetic)

    return(list_synthetic)

def plotting_triangle_delimiter_and_fix_size(ax,title,plotwidth=300):
    list_triplets=[[1,0,0],[0,1,0],[0,0,1]]
    list_labels=['FC','CT','FD']
    list_labels_pos=[(-0.235*plotwidth,-0.012*plotwidth),(-0.08*plotwidth,0.022*plotwidth),(0.008*plotwidth,-0.014*plotwidth)]
    trianglex=[]
    triangley=[]
    for index,i in enumerate(list_triplets):
        x,y=triple2ternary(i, plotwidth)
        trianglex.append(x)
        triangley.append(y)
        plt.annotate(list_labels[index],xy=(x,y),xytext=(x+list_labels_pos[index][0],y+list_labels_pos[index][1]),
                     fontsize=10)

    trianglex.append(trianglex[0])
    triangley.append(triangley[0])

    for i in range(3):
        plt.plot(trianglex, triangley, '-',color='k',alpha=1,lw=0.5,zorder=1)
    plt.fill(trianglex, triangley,c='gray',alpha=0.1,zorder=1)
#     plt.xlim(0,1.0714*plotwidth)

#     plt.ylim(0.928*plotwidth,1.928*plotwidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    plt.xticks([])
    
##Setting the colors:
def CLM_color(flag=1):#1->Chaos,2->PS, 3->Intermittency, 4->Random
    if flag==1:
        list_of_colors=sns.color_palette('Blues',n_colors=12)
        list_of_colors=list_of_colors[3:]
    if flag==2:
        list_of_colors=sns.color_palette('YlOrBr',n_colors=12)
        list_of_colors=list_of_colors[4:]
    if flag==3:
        list_of_colors=sns.color_palette('Greens',n_colors=12)
        list_of_colors=list_of_colors[4:]
    if flag==4:
        list_of_colors=sns.color_palette('Reds',n_colors=12)
        list_of_colors=list_of_colors[4:]
    if flag==5:
        list_of_colors=sns.color_palette('Purples',n_colors=12)
        list_of_colors=list_of_colors[4:]        
    if flag==6:
        list_of_colors=sns.color_palette('copper',n_colors=12)
        list_of_colors=list_of_colors[2:]
    if flag==7:
        list_of_colors=sns.color_palette('RdPu',n_colors=12)
        list_of_colors=list_of_colors[4:]
    if flag==8:
        list_of_colors=sns.color_palette('cividis',n_colors=10)
        list_of_colors=list_of_colors[4:]
    my_cmap = ListedColormap(list_of_colors, name='from_list', N=None)
    return(my_cmap)




## Evaluate the network partition using element-centric similarity 
def evaluate_partition(partition):
    c1 = Clustering(elm2clu_dict={nodeID: [idx] for idx, cluster in enumerate(
        partition) for nodeID in cluster})
    box_cumsum = np.cumsum([0, 300, 144, 156, 273, 170, 138, 128, 312])
    ground_truth = Clustering(elm2clu_dict={nodeID: [idx] for idx in range(
        len(box_cumsum)-1) for nodeID in range(box_cumsum[idx], box_cumsum[idx+1])})
    return(sim.element_sim(c1, ground_truth, alpha=0.9))


### Plotting function for adjusting the ticks and axes width
def plot_adjustments(ax,ticks_width=1.5,axis_width=2.5,length=2,upper_right=False,labelsize=10):
    plt.rcParams['font.family'] = "PT Serif Caption"
    plt.rcParams['xtick.major.width'] = ticks_width
    plt.rcParams['ytick.major.width'] = ticks_width
    plt.rcParams['axes.linewidth'] = ticks_width
    if upper_right==True:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False) 
    ax.spines["top"].set_linewidth(axis_width)
    ax.spines["right"].set_linewidth(axis_width)
    ax.spines["left"].set_linewidth(axis_width)
    ax.spines["bottom"].set_linewidth(axis_width)
    ax.tick_params(direction='out', length=4, width=axis_width, colors='k',
           grid_color='k', grid_alpha=0.5,axis='both')
    ax.tick_params(direction='out',which='minor', length=length, width=axis_width, colors='k',
           grid_color='k', grid_alpha=0.5,axis='both')
    ax.tick_params(direction='out',which='major', length=length, colors='k',
           grid_color='k', grid_alpha=0.5,axis='both')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    #ax.set_xlim(-50,1200)


            
            
label_list = ['REST', 'EMO', 'GAM', 'LAN', 'MOT', 'REL', 'SOC', 'WM'] # List of task names
box_cumsum = np.cumsum([0, 300, 144, 156, 273, 170, 138, 128, 312])+1 # Indexes of the tasks in the time series
map_label_intervals={label_list[i]:[box_cumsum[i]-1,box_cumsum[i+1]-1] for i in range(len(label_list))}



### Loading the HO indicators for the different tasks
data_rest=[]
data_motor=[]
data_language=[]
data_emotion=[]
data_wm=[]
data_gambling=[]
data_relational=[]
data_social=[]
for pathdata in sorted(glob.glob('/home/andrea/miplabsrv2/HCP/*')):
    subjID=pathdata.split('/')[-1]
#     print(subjID,"LR")
    pathdata='./../../Results/HO_indicators/HOindicators_%s_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_LR' % subjID
    data_all_current=all_task_file(pathdata,map_label_intervals)
    data_rest.append(data_all_current['REST'])
    data_motor.append(data_all_current['MOT'])
    data_language.append(data_all_current['LAN'])
    data_emotion.append(data_all_current['EMO'])
    data_wm.append(data_all_current['WM'])
    data_gambling.append(data_all_current['GAM'])
    data_relational.append(data_all_current['REL'])
    data_social.append(data_all_current['SOC'])
    
#     print(subjID,"RL")
    pathdata='./../../Results/HO_indicators/HOindicators_%s_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_RL' % subjID
    data_all_current=all_task_file(pathdata,map_label_intervals)
    data_rest.append(data_all_current['REST'])
    data_motor.append(data_all_current['MOT'])
    data_language.append(data_all_current['LAN'])
    data_emotion.append(data_all_current['EMO'])
    data_wm.append(data_all_current['WM'])
    data_gambling.append(data_all_current['GAM'])
    data_relational.append(data_all_current['REL'])
    data_social.append(data_all_current['SOC'])
    
    

## Create dataframes with all the data
column_used=5
all_data=[data_rest,data_emotion,data_gambling,data_language,
          data_motor,data_relational,data_social,data_wm]


all_sampling=[]
all_label=[]

all_sampling_scatter=[]
all_label_scatter=[]

###Preparing the two vectors all_sampling and all_label to create a dataframe: Inserting the CML data
sampling_size=100000
sampling_size_for_scatter=500    
# data_label=['Rest','Motor','Language', 'Emotion','WM','Gambling','Relational','Social']
data_label=['Rest','Emotion','Gambling','Language','Motor','Relational','Social','WM']

for index_data,current_data in enumerate(all_data): 

    current_column=current_data[0][:,column_used]
    print(index_data,data_label[index_data],np.shape(current_column),len(current_data))
#     print(index_data,np.shape(current_column))
    for i in range(1,len(current_data)):
        current_column=np.concatenate((current_column,current_data[i][:,column_used]))
    sampling_current_column=np.random.choice(current_column,sampling_size)
    current_label=[data_label[index_data] for i in range(sampling_size)]
    all_sampling.extend(sampling_current_column)
    all_label.extend(current_label)
    
    
    sampling_current_column_scatter=np.random.choice(sampling_current_column,sampling_size_for_scatter)
    current_label_scatter=[data_label[index_data] for i in range(sampling_size_for_scatter)]
    all_sampling_scatter.extend(sampling_current_column_scatter)
    all_label_scatter.extend(current_label_scatter)

prova_data = {'Hypercoherence': all_sampling, 'type': all_label}
prova_data=pd.DataFrame.from_dict(prova_data)


prova_data_scatter = {'Hypercoherence': all_sampling_scatter, 'type': all_label_scatter}
prova_data_scatter=pd.DataFrame.from_dict(prova_data_scatter)




## Plotting the first higher-order plots

fig=plt.figure(figsize=(12,12.5),dpi=300)
# gs = GridSpec(8, 9, figure=fig,height_ratios = [0.55,0.55,0.55,1.45,0.25,0.25,0.25,0.25,], wspace=0.3)
gs = GridSpec(6, 8, figure=fig,wspace=0.55,hspace=0.65,height_ratios=[0.35,0.35,0.1,1,1,1])
#ax=plt.subplot(211)
ax=fig.add_subplot(gs[0:2, 0:4])

# plot_adjustments(ax,ticks_width=1.25,axis_width=1.25)
sns.set(style="ticks")
# sns.set_theme(style="whitegrid")

ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  


color_chaos='#88419d'
color_PS='#ec7014'
color_STI='#41ab5d'
color_BMWD='#1f78b4'
color_DT='#e31a1c'
color_nullmodel='#bbbbbb80'
color_brain='#08519c'
color_financial='#ef3b2c'
color_social='#4eb3d3e5'
color_influenza='#c7e9c0ff'
color_gonorrhea='#fdae6bff'
color_measles='#dadaebff'


colors = [color_chaos, color_PS,color_STI,color_BMWD,color_DT,color_nullmodel,
         color_brain,color_financial,
          #color_social,color_nullmodel]# Set your custom color palette
          color_influenza,color_gonorrhea,
          color_measles]
palette=sns.color_palette(colors)

ax = sns.violinplot(x="type", y="Hypercoherence", data=prova_data,
                    palette=palette,
                    scale="width", inner=None,clip_on=True,cut=0)

alpha_level=1
for violin in ax.collections:
    violin.set_alpha(alpha_level)

old_len_collections = len(ax.collections)
print(old_len_collections)
for t in range(0,old_len_collections):
    offset = transforms.ScaledTranslation(-0.05, 0, ax.figure.dpi_scale_trans)
    trans = ax.collections[t].get_transform()
    ax.collections[t].set_transform(trans + offset)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0-0.12*width, y0), width/2-0.00*width, height, transform=ax.transData))



PROPS = {
    'boxprops':{ 'edgecolor':'#e31a1ccc','zorder': 5,'facecolor':'none'},
    'medianprops':{'color':'#e31a1ccc','zorder':5},
    'whiskerprops':{'color':'#e31a1ccc','zorder':5},
    'capprops':{'color':'#e31a1ccc','zorder':5},
    
}

sns.boxplot(x="type", y="Hypercoherence", data=prova_data, saturation=1, showfliers=False,
             width=0.1,ax=ax,**PROPS,linewidth=1)

ax.spines["bottom"].set_visible(False)
# # sns.despine(offset=10, trim=True)

sns.stripplot(x="type", y="Hypercoherence", data=prova_data_scatter, color='gray', ax=ax,size=3,jitter=0.01,alpha=0.05)


ax.tick_params(axis='x', rotation=30)
xset=0.2
plt.ylim(xset,1)
plt.xlim(-0.6,8)
plt.ylabel("")
plt.xlabel("")
plt.xticks(ha='right')
# plt.hlines(xset,0,5,color='k')
# plt.hlines(xset,7,8,color='k')
# plt.hlines(xset,10,11,color='k')
# plt.hlines(xset,13,18,color='k')
for ss in range(len(ax.collections)):
    ax.collections[ss].set_edgecolor((0,0,0, 1))
    
plt.ylabel('Hyper-coherence',labelpad=10,fontsize=12)
ax.set_xlim(-0.7,8)

plt.text(x=-1.8,y=1.1,s="a",clip_on=False,weight='bold')

############################################################Top right plot


from matplotlib import colors
color_chaos='#88419d'
color_PS='#ec7014'
color_STI='#41ab5d'
color_BMWD='#1f78b4'
color_DT='#e31a1c'
end=100000
al=0.5
al_hist=0.8
# fig=plt.figure(figsize=(11,4.48),dpi=300)

row=0
column=4
for idx_recording,data_current in enumerate(all_data):
    print(data_label[idx_recording],row, column+idx_recording%4)
    if row == 0:
        ax=fig.add_subplot(gs[0, column+idx_recording%4])
        if idx_recording==0:
            plt.text(x=-100,y=(1+np.sqrt(3)/2)*300+120,s="b",clip_on=False,weight='bold')
    else:
        ax=fig.add_subplot(gs[1, column+idx_recording%4])
#     ax=plt.subplot(int('24{0}'.format(idx_recording+1)))
#     plot_adjustments(ax)
    plotting_triangle_delimiter_and_fix_size(ax,'CLM')
    list_synthetic=extract_triplets_brain(data_current)
#     plt.scatter(list_synthetic[0,0],list_synthetic[0,1],color=color_chaos,marker='v',s=2,alpha=0.5,label='Chaos (FDT)')
    plt.hist2d(list_synthetic[:,0], list_synthetic[:,1],bins=60, norm=colors.LogNorm(), cmap=CLM_color(flag=idx_recording+1),zorder=50,alpha=al_hist);
    plt.xlim(0,300)
    plt.ylim(300,(1+np.sqrt(3)/2)*300)
    if row == 0:
        plt.title(data_label[idx_recording],fontsize=12,pad=15)
    else:
#         ax.annotate(,fontsize=12,pad=15)
        if column+idx_recording%4>=6: 
            ax.text(0.55-0.048*len(data_label[idx_recording]), -0.4, data_label[idx_recording], transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom')
        else:
            ax.text(0.5-0.055*len(data_label[idx_recording]), -0.4, data_label[idx_recording], transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom')
        
    if idx_recording==3:
        row+=1
        
        



### Loading the recurrence plots
similarity_BOLD_LR=np.load('./../../Results/HO_recurrence_plot/similarity_BOLD_0_100_zscored_LR.npy')
similarity_BOLD_RL=np.load('./../../Results/HO_recurrence_plot/similarity_BOLD_0_100_zscored_RL.npy')

similarity_edges_LR=np.load('./../../Results/HO_recurrence_plot/similarity_edges_0_100_zscored_LR.npy')
similarity_edges_RL=np.load('./../../Results/HO_recurrence_plot//similarity_edges_0_100_zscored_RL.npy')

similarity_triangles_LR=np.load('./../../Results/HO_recurrence_plot/similarity_triangles_0_100_zscored_LR.npy')
similarity_triangles_RL=np.load('./../../Results/HO_recurrence_plot/similarity_triangles_0_100_zscored_RL.npy')

similarity_scaffold_LR=np.load('./../../Results/HO_recurrence_plot/similarity_scaffold_0_100_zscored_LR_frequency.npy')
similarity_scaffold_RL=np.load('./../../Results/HO_recurrence_plot/similarity_scaffold_0_100_zscored_RL_frequency.npy')


similarity_BOLD = similarity_BOLD_LR + similarity_BOLD_RL
similarity_edges = similarity_edges_LR + similarity_edges_RL
similarity_triangles = similarity_triangles_LR + similarity_triangles_RL
similarity_scaffold = similarity_scaffold_LR + similarity_scaffold_RL




### Performing the Community detection based on the binarized recurrence matrix
max_iteration=10
perc=95

#1. Performance of BOLD
print("Computing BOLD Community score...")
G=nx.from_numpy_array(np.where(similarity_BOLD>np.percentile(similarity_BOLD,perc),1,0)-np.eye(1621))
BOLD_communities_Pearson=np.mean([evaluate_partition(nx.algorithms.community.louvain_communities(G)) for i in range(max_iteration)])


#2. Performance of the ETS
print("Computing Edge Community score...")
G=nx.from_numpy_array(np.where(similarity_edges>np.percentile(similarity_edges,perc),1,0)-np.eye(1621))
Edges_communities_Pearson=np.mean([evaluate_partition(nx.algorithms.community.louvain_communities(G)) for i in range(max_iteration)])

#3. Performance of Triangles

print("Computing Triangle Community score...")
G=nx.from_numpy_array(np.where(similarity_triangles>np.percentile(similarity_triangles,perc),1,0)-np.eye(1621))
triangles_communities_Pearson=np.mean([evaluate_partition(nx.algorithms.community.louvain_communities(G)) for i in range(max_iteration)])

#4. Performance of the Scaffold (Frequency)

print("Computing Scaffold Community score...")
G=nx.from_numpy_array(np.where(similarity_scaffold>np.percentile(similarity_scaffold,perc),1,0)-np.eye(1621))
scaffold_communities_Pearson=np.mean([evaluate_partition(nx.algorithms.community.louvain_communities(G)) for i in range(max_iteration)])



fig=plt.figure(figsize=(12,6),dpi=150)
fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(12,6),dpi=150)

cmap='twilight_shifted'
alphaval=1
min_val=5
max_val=95
perc=95
u,v=np.triu_indices(len(similarity_BOLD),k=1,m=len(similarity_BOLD))

ax= axes[0]
ax.matshow(np.where(similarity_BOLD>np.percentile(similarity_BOLD,perc),similarity_BOLD,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_BOLD,95),vmax=np.percentile(similarity_BOLD,99))
           #cmap='twilight_shifted', norm=LogNorm(vmin=np.percentile(similarity_BOLD,perc), vmax=np.percentile(similarity_BOLD,99)))
# plot_adjustments(ax=ax,ticks_width=1.25,axis_width=1.25)

ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
ax.set_xticks([0,400,800,1200,1600])
ax.set_yticks([0,400,800,1200,1600])
ax.set_ylabel('Time',fontsize=12.5,labelpad=3)
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)
plot_boxes(color='darkred',lw=1.,ax=ax)
ax.set_title('BOLD time signal',fontsize=14)
# plot_boxes()
for label,label1 in zip(ax.get_yticklabels(),ax.get_xticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(10)
    label1.set_fontsize(10)
# ax.spines("top").set_visible("False")
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 3 )

################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (BOLD_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################
    
    
ax= axes[1]
ax.matshow(np.where(similarity_edges>np.percentile(similarity_edges,perc),similarity_edges,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_edges,95),vmax=np.percentile(similarity_edges,99))


ax.set_yticks([])
ax.set_xticks([0,400,800,1200,1600])

ax.set_xlabel('Time',fontsize=12.5,labelpad=3)
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
plot_boxes(color='darkred',lw=1.,ax=ax)
# plot_boxes()
ax.set_title('Edge time signal',fontsize=14)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 0 )


################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (Edges_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################



ax= axes[2]
ax.matshow(np.where(similarity_triangles>np.percentile(similarity_triangles,perc),similarity_triangles,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_triangles,95),vmax=np.percentile(similarity_triangles,99))
           #cmap='twilight_shifted', norm=LogNorm(vmin=np.percentile(similarity_triangles,perc), vmax=np.percentile(similarity_triangles,99)))
# ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks([0,400,800,1200,1600])
# plt.title('Triangles HO')
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
plot_boxes(color='darkred',lw=1.,ax=ax)
ax.set_title('Triangle time signal',fontsize=14)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 45,
    pad = 0 )
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)


################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (triangles_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################


ax= axes[3]
# ax.matshow(np.where(similarity_scaffold>np.percentile(similarity_scaffold,perc),similarity_scaffold,np.nan),cmap='Greys_r')
im=ax.matshow(np.where(similarity_scaffold>np.percentile(similarity_scaffold,perc),similarity_scaffold,np.nan),
           cmap=cmap, vmin=np.percentile(similarity_scaffold,95),vmax=np.percentile(similarity_scaffold,99))

# ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
ax.set_xticks([0,400,800,1200,1600])

plot_boxes(color='darkred',lw=1.,ax=ax)
# t = [1.0,2.0,5.0,10.0]
# fig.colorbar(im, format="$%.2f$",fraction=0.046)
ax.set_title('Scaffold time signal',fontsize=14)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad =3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 0 )
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)

################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (scaffold_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################

list_tasks=['Rest','Emotion','Gambling','Language','Motor','Relational','Social','WM']
list_colors=[i if idx!=7 else sns.color_palette("tab10", 10)[8] for idx,i in enumerate(sns.color_palette("tab10", 10)[:len(list_tasks)])]

legend_elements=[Patch(facecolor='white', edgecolor=i,
                         label=list_tasks[idx],linewidth=1.5) for idx,i in enumerate(list_colors)]

plt.subplots_adjust(wspace=0.1,hspace=0.05)

# for axes in ax[:3]:
#     axes.set_aspect(10)

# cax = ax
# plt.colorbar(cax=cax)

# # plt.legend(handles=handles, labels=colors, handler_map=hmap)

# cbar = fig.colorbar(im,ax=axes.ravel().tolist(),fraction=0.0115,pad=0.01,ticks=[4, 5, 6])
# cbar.set_ticklabels([95,97,99])
# cbar.set_label("Percentile", labelpad=-1)
# cbar.ax.yaxis.set_tick_params(pad=3)

plt.legend(bbox_to_anchor=(1.05,-0.2),handles=legend_elements,ncol=10,fontsize=10)
#b
plt.tight_layout()
plt.savefig('SI_fig1_1.png',dpi=600)




#### Combining the figure in one plot


fig=plt.figure(figsize=(12,12.5),dpi=300)
# gs = GridSpec(8, 9, figure=fig,height_ratios = [0.55,0.55,0.55,1.45,0.25,0.25,0.25,0.25,], wspace=0.3)
gs = GridSpec(6, 8, figure=fig,wspace=0.55,hspace=0.65,height_ratios=[0.35,0.35,0.1,1,1,1])
#ax=plt.subplot(211)
ax=fig.add_subplot(gs[0:2, 0:4])

# plot_adjustments(ax,ticks_width=1.25,axis_width=1.25)
sns.set(style="ticks")
# sns.set_theme(style="whitegrid")

ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  


color_chaos='#88419d'
color_PS='#ec7014'
color_STI='#41ab5d'
color_BMWD='#1f78b4'
color_DT='#e31a1c'
color_nullmodel='#bbbbbb80'
color_brain='#08519c'
color_financial='#ef3b2c'
color_social='#4eb3d3e5'
color_influenza='#c7e9c0ff'
color_gonorrhea='#fdae6bff'
color_measles='#dadaebff'


colors = [color_chaos, color_PS,color_STI,color_BMWD,color_DT,color_nullmodel,
         color_brain,color_financial,
          #color_social,color_nullmodel]# Set your custom color palette
          color_influenza,color_gonorrhea,
          color_measles]
palette=sns.color_palette(colors)

ax = sns.violinplot(x="type", y="Hypercoherence", data=prova_data,
                    palette=palette,
                    scale="width", inner=None,clip_on=True,cut=0)

alpha_level=1
for violin in ax.collections:
    violin.set_alpha(alpha_level)

old_len_collections = len(ax.collections)
print(old_len_collections)
for t in range(0,old_len_collections):
    offset = transforms.ScaledTranslation(-0.05, 0, ax.figure.dpi_scale_trans)
    trans = ax.collections[t].get_transform()
    ax.collections[t].set_transform(trans + offset)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    bbox = violin.get_paths()[0].get_extents()
    x0, y0, width, height = bbox.bounds
    violin.set_clip_path(plt.Rectangle((x0-0.12*width, y0), width/2-0.00*width, height, transform=ax.transData))



PROPS = {
    'boxprops':{ 'edgecolor':'#e31a1ccc','zorder': 5,'facecolor':'none'},
    'medianprops':{'color':'#e31a1ccc','zorder':5},
    'whiskerprops':{'color':'#e31a1ccc','zorder':5},
    'capprops':{'color':'#e31a1ccc','zorder':5},
    
}

sns.boxplot(x="type", y="Hypercoherence", data=prova_data, saturation=1, showfliers=False,
             width=0.1,ax=ax,**PROPS,linewidth=1)

ax.spines["bottom"].set_visible(False)
# # sns.despine(offset=10, trim=True)

sns.stripplot(x="type", y="Hypercoherence", data=prova_data_scatter, color='gray', ax=ax,size=3,jitter=0.01,alpha=0.05)


ax.tick_params(axis='x', rotation=30)
xset=0.2
plt.ylim(xset,1)
plt.xlim(-0.6,8)
plt.ylabel("")
plt.xlabel("")
plt.xticks(ha='right')
# plt.hlines(xset,0,5,color='k')
# plt.hlines(xset,7,8,color='k')
# plt.hlines(xset,10,11,color='k')
# plt.hlines(xset,13,18,color='k')
for ss in range(len(ax.collections)):
    ax.collections[ss].set_edgecolor((0,0,0, 1))
    
plt.ylabel('Hyper-coherence',labelpad=10,fontsize=12)
ax.set_xlim(-0.7,8)

plt.text(x=-1.8,y=1.1,s="a",clip_on=False,weight='bold')

############################################################Top right plot


from matplotlib import colors
color_chaos='#88419d'
color_PS='#ec7014'
color_STI='#41ab5d'
color_BMWD='#1f78b4'
color_DT='#e31a1c'
end=100000
al=0.5
al_hist=0.8
# fig=plt.figure(figsize=(11,4.48),dpi=300)

row=0
column=4
for idx_recording,data_current in enumerate(all_data):
    print(data_label[idx_recording],row, column+idx_recording%4)
    if row == 0:
        ax=fig.add_subplot(gs[0, column+idx_recording%4])
        if idx_recording==0:
            plt.text(x=-100,y=(1+np.sqrt(3)/2)*300+120,s="b",clip_on=False,weight='bold')
    else:
        ax=fig.add_subplot(gs[1, column+idx_recording%4])
#     ax=plt.subplot(int('24{0}'.format(idx_recording+1)))
#     plot_adjustments(ax)
    plotting_triangle_delimiter_and_fix_size(ax,'CLM')
    list_synthetic=extract_triplets_brain(data_current)
#     plt.scatter(list_synthetic[0,0],list_synthetic[0,1],color=color_chaos,marker='v',s=2,alpha=0.5,label='Chaos (FDT)')
    plt.hist2d(list_synthetic[:,0], list_synthetic[:,1],bins=60, norm=colors.LogNorm(), cmap=CLM_color(flag=idx_recording+1),zorder=50,alpha=al_hist);
    plt.xlim(0,300)
    plt.ylim(300,(1+np.sqrt(3)/2)*300)
    if row == 0:
        plt.title(data_label[idx_recording],fontsize=12,pad=15)
    else:
#         ax.annotate(,fontsize=12,pad=15)
        if column+idx_recording%4>=6: 
            ax.text(0.55-0.048*len(data_label[idx_recording]), -0.4, data_label[idx_recording], transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom')
        else:
            ax.text(0.5-0.055*len(data_label[idx_recording]), -0.4, data_label[idx_recording], transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom')
        
    if idx_recording==3:
        row+=1


    
    
    
##############################################################Bottom plots##########################################

cmap='twilight_shifted'
alphaval=1
min_val=5
max_val=95
perc=95
u,v=np.triu_indices(len(similarity_BOLD),k=1,m=len(similarity_BOLD))

ax=fig.add_subplot(gs[2:4,0:2])
ax.matshow(np.where(similarity_BOLD>np.percentile(similarity_BOLD,perc),similarity_BOLD,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_BOLD,95),vmax=np.percentile(similarity_BOLD,99))
plt.text(x=-450,y=-200,s="c",clip_on=False,weight='bold')
           #cmap='twilight_shifted', norm=LogNorm(vmin=np.percentile(similarity_BOLD,perc), vmax=np.percentile(similarity_BOLD,99)))
# plot_adjustments(ax=ax,ticks_width=1.25,axis_width=1.25)

ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
ax.set_xticks([0,400,800,1200,1600])
ax.set_yticks([0,400,800,1200,1600])
ax.set_ylabel('Time',fontsize=12.5,labelpad=0)
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)
plot_boxes(color='darkred',lw=1.2,ax=ax)
ax.set_title('BOLD time signal',fontsize=14)
# plot_boxes()
for label,label1 in zip(ax.get_yticklabels(),ax.get_xticklabels()):
    #label.set_fontname('Arial')
    label.set_fontsize(12)
    label1.set_fontsize(12)
# ax.spines("top").set_visible("False")
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 3 )

################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (BOLD_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat',edgecolor='darkgray', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.58, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################
    
    
# ax=fig.add_subplot(gs[2:4, 2:])
ax=fig.add_subplot(gs[2:4,2:4])
ax.matshow(np.where(similarity_edges>np.percentile(similarity_edges,perc),similarity_edges,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_edges,95),vmax=np.percentile(similarity_edges,99))


ax.set_yticks([])
ax.set_xticks([0,400,800,1200,1600])

ax.set_xlabel('Time',fontsize=12.5,labelpad=3)
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
plot_boxes(color='darkred',lw=1.2,ax=ax)
# plot_boxes()
ax.set_title('Edge time signal',fontsize=14)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 0 )




################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (Edges_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat',edgecolor='darkgray', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.58, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################



ax=fig.add_subplot(gs[2:4,4:6])
ax.matshow(np.where(similarity_triangles>np.percentile(similarity_triangles,perc),similarity_triangles,np.nan),
           cmap=cmap,vmin=np.percentile(similarity_triangles,95),vmax=np.percentile(similarity_triangles,99))
           #cmap='twilight_shifted', norm=LogNorm(vmin=np.percentile(similarity_triangles,perc), vmax=np.percentile(similarity_triangles,99)))
# ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks([0,400,800,1200,1600])
# plt.title('Triangles HO')
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
plot_boxes(color='darkred',lw=1.2,ax=ax)
ax.set_title('Triangle time signal',fontsize=14)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad = 3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 45,
    pad = 0 )
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)


################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (triangles_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat',edgecolor='darkgray', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.58, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################


ax=fig.add_subplot(gs[2:4,6:8])
# ax.matshow(np.where(similarity_scaffold>np.percentile(similarity_scaffold,perc),similarity_scaffold,np.nan),cmap='Greys_r')
im=ax.matshow(np.where(similarity_scaffold>np.percentile(similarity_scaffold,perc),similarity_scaffold,np.nan),
           cmap=cmap, vmin=np.percentile(similarity_scaffold,95),vmax=np.percentile(similarity_scaffold,99))

# ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,1621)
ax.set_ylim(1621,0)
ax.set_xticks([0,400,800,1200,1600])

plot_boxes(color='darkred',lw=1.2,ax=ax)
# t = [1.0,2.0,5.0,10.0]
# fig.colorbar(im, format="$%.2f$",fraction=0.046)
ax.set_title('Scaffold time signal',fontsize=14)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True,  # labels along the bottom edge are off
    labeltop=False,
    rotation = 0,
    pad =3)

ax.tick_params(
    axis='y',          # changes apply to the y-axis
    rotation = 0,
    pad = 0 )
ax.set_xlabel('Time',fontsize=12.5,labelpad=3)

################Text Annotation##########################
textstr = r'$ECS=%.2f$' % (scaffold_communities_Pearson)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat',edgecolor='darkgray', alpha=0.3)

# place a text box in upper left in axes coords
ax.text(0.58, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
#########################################################

list_tasks=['Rest','Emotion','Gambling','Language','Motor','Relational','Social','WM']
list_colors=[i if idx!=7 else sns.color_palette("tab10", 10)[8] for idx,i in enumerate(sns.color_palette("tab10", 10)[:len(list_tasks)])]

legend_elements=[Patch(facecolor='white', edgecolor=i,
                         label=list_tasks[idx],linewidth=1.5) for idx,i in enumerate(list_colors)]

# plt.subplots_adjust(wspace=0.1,hspace=0.05)

# for axes in ax[:3]:
#     axes.set_aspect(10)

# cax = ax
# plt.colorbar(cax=cax)

# # plt.legend(handles=handles, labels=colors, handler_map=hmap)
cbar = fig.colorbar(im,ax=axes.ravel().tolist(),fraction=0.0115,pad=0.01,ticks=[4, 5, 6])
cbar.set_ticklabels([95,97,99])
cbar.set_label("Percentile", labelpad=-1)
cbar.ax.yaxis.set_tick_params(pad=3)

plt.legend(bbox_to_anchor=(1.15,-0.22),handles=legend_elements,ncol=10,fontsize=11)
plt.savefig('fig2_final.svg',dpi=150,transparent=True,bbox_inches='tight')


