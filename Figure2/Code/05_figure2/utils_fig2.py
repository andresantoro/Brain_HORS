import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math 
from matplotlib.colors import ListedColormap
from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim



## Plot boxes for the HCP cognitive tasks
def plot_boxes(color='red', lw=2, linestyles='solid', ax=None):
    ''' 
    Function that plots the boxes corresponding to the HCP tasks
    in the recurrence plots 
    '''
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
            
    
def all_task_file(filename,map_label_intervals):
    '''
    Function that loads the data containing the HO indicators

    Parameters
    ----------
    filename : name of the file containing the HO indicators
    map_label_intervals : dictionary with keys corresponding to task names
    and for values a tuple containing starting and end of the HCP task

    Returns
    -------
    Dictionary with task names as key and the HO indicators as values

    '''
    
    data=np.array(sorted(np.loadtxt(filename),key=lambda x:x[0]))
    data_all={i:[] for i in map_label_intervals}
    for task in map_label_intervals:
        idx1,idx2=map_label_intervals[task][0],map_label_intervals[task][1]
        data_all[task]=data[idx1:idx2,:]
    return(data_all)




def triple2ternary(triple, pw=1):
    '''
    Function that maps a 3-tuple into the 2-dimensional coordinate within a triangle  

    Parameters
    ----------
    triple : Tuple of three values
    pw : Rescaling factor (plot width) for the 2-D coordinate within the triangle
        TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    A 2-D coordinate in the triangles

    '''
    # normalize 
    triple = [float(i)/sum(triple) for i in triple]
    y = triple[1] * math.sin(math.radians(60));
    x = 1 - triple[0] - y * 1/math.tan(math.radians(60));
    y = 1+y
    return pw*x,pw*y

def extract_triplets_group(alldata,plotwidth=300):
    '''
    Function that from a list of lists of HO indicators takes the complexity
    contributes (columns 2-5) and return for each point the corresponding
    2-D coordinate within a triangle

    Parameters
    ----------
    triple : list of lists of HO indicators, each list should have at least 
             5 columns (the complexity contributes are located in cols 2-5)
    pw : Rescaling factor (plot width) for the 2-D coordinate within the triangle
        TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    A numpy array containing the 2-D coordinates in a triangle

    '''
    current_triplet=alldata[0][:,2:5]
    for i in range(1,len(alldata)):
        current_behaviour=alldata[i][:,2:5]
        current_triplet=np.concatenate((current_triplet,current_behaviour))
    list_synthetic=[]
    for i in current_triplet:
        x,y=triple2ternary(i, plotwidth)
        list_synthetic.append([x,y])
    list_synthetic=np.array(list_synthetic)

    return(list_synthetic)

def plotting_triangle_delimiter_and_fix_size(ax,title,plotwidth=300):
    '''
    Plotting a gray triangle with the corresponding title

    Parameters
    ----------
    ax : matplotlib ax for subplots
    title : title of the plot/subplot
    
    plotwidth : TYPE, optional
        DESCRIPTION. The default is 300.

    Returns
    -------
    None.

    '''
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
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    plt.xticks([])
    
    

def CLM_color(flag=1):
    '''
    Function that generates different color palettes according to the
    flag given in input

    Parameters
    ----------
    flag : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    ListedColormap of the color palette

    '''
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



def evaluate_partition(partition, box_cumsum = np.cumsum([0, 300, 144, 156, 273, 170, 138, 128, 312])):
    '''
    Evaluate the community partition of a network against a ground truth
    using element-centric similarity 


    Parameters
    ----------
    partition : Partition of a network, 
        
    box_cumsum : list of indexes representening the position of the blocks

    Returns
    -------
    Element-centric similarity for the partition inserted

    '''
    c1 = Clustering(elm2clu_dict={nodeID: [idx] for idx, cluster in enumerate(
        partition) for nodeID in cluster})
    
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