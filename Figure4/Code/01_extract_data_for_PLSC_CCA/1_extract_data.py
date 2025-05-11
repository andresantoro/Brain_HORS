#!/usr/bin/env python3

## Importing libraries
import tqdm
import sys 
import numpy as np
from itertools import combinations
from scipy.io import loadmat,savemat
import glob
import pickle as pk
import os
import h5py
import pickle as pkl

def upper_tri_masking(A):
    '''Extract the upper triangular
    values of a matrix'''
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]  

def compute_MAD_and_extract_data(X,thresh=90):
    MAD=np.median(np.abs(X-np.median(X,axis=0)),axis=0),
    indices_MAD=np.where(MAD >=np.percentile(MAD,thresh))[1]
    return X[:,indices_MAD]

def compute_MAD_and_extract_data_with_exact_number(X,features=5):
    MAD=np.median(np.abs(X-np.median(X,axis=0)),axis=0)
    MADmax=sorted(MAD)[-features]
    indices_MAD=np.where(MAD >= MADmax)[0]
    return X[:,indices_MAD]

def compute_PCA_and_extract_data(X,ncomps=10):
    pca = PCA(n_components=ncomps)
    # prepare transform on dataset
    pca.fit(X)
    # apply transform to dataset
    transformed = pca.transform(X)
    print("Explained variance (%): ", pca.explained_variance_ratio_[:ncomps].sum()*100)
    return(transformed)

def loading_yeo(path='../../../Misc/yeo_RS7_Schaefer100S.mat'):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN','SC','C'])}
    yeoROI_dict['SC']=np.array(sorted(np.hstack((yeoROI_dict['SC'],yeoROI_dict['C']))))
    del yeoROI_dict['C']
    return(yeoROI_dict)


def compute_eFC(data_TS):
    '''Compute the Edge Functional connectivty
    starting from a BOLD time series'''
    N,T=np.shape(data_TS)
    u,v=np.triu_indices(N,k=1,m=N)
    data_eFC=data_TS[u,:]*data_TS[v,:]
    return(np.corrcoef(data_eFC))




##Loading data
N_nodes=119
N_subjects=100
map_idx_edges={val:idx for idx,val in enumerate(combinations(np.arange(N_nodes),2))}
map_idx_triangles={val:idx for idx,val in enumerate(combinations(np.arange(N_nodes),3))}
N_edges=len(map_idx_edges)
N_triangles=len(map_idx_triangles)
yeo_dict=loading_yeo()
yeoROIs=np.array([i[0]-1 for i in loadmat('../../../Misc/yeo_RS7_Schaefer100S.mat')['yeoROIs']]) ## Loading the data
path_HCP_data="/storage/Projects/HCP_MIPLAB/results_v2/"
path_output_data="./../../Results/brain_features/"


#### Load BOLD data and compute the functional connectivity (averaged over day1-day2)
brain_data=[]
for path_subjID in tqdm.tqdm(sorted(glob.glob(path_HCP_data+'*'))):
    subjID=path_subjID.strip().split('/')[-1]
    input_folder1=f'{path_HCP_data}{subjID}/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
    input_folder2=f'{path_HCP_data}{subjID}/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
    data1=upper_tri_masking(np.corrcoef(loadmat(input_folder1)['TS']))
    data2=upper_tri_masking(np.corrcoef(loadmat(input_folder2)['TS']))
    brain_features=0.5*(data1+data2)
    brain_data.append(brain_features)
brain_data=np.array(brain_data)


#### Save brain FC data and also the MAD with only 10 features
ncomponents=10
X=brain_data
mdic={'X':X}
savemat(path_output_data+"X_BOLD_ALL.mat",mdic)

X=compute_MAD_and_extract_data_with_exact_number(brain_data,features=ncomponents)
mdic={'X':X}
savemat(path_output_data+f'X_BOLD_ALL_MAD_{ncomponents}.mat',mdic)




### Load BOLD data and compute edge functional connectivity
# u,v = np.triu_indices(n=N_nodes,k=1,m=N_nodes)
# eFC=[]
# for path_subjID in tqdm.tqdm(sorted(glob.glob(path_HCP_data+'*'))):
#     subjID=path_subjID.strip().split('/')[-1]
#     input_folder1=f'{path_HCP_data}{subjID}/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
#     input_folder2=f'{path_HCP_data}{subjID}/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
#     ts1=loadmat(input_folder1)['TS']
#     ts2=loadmat(input_folder2)['TS']
#     data1=upper_tri_masking(np.corrcoef(ts1[u,:]*ts1[v,:]))
#     data2=upper_tri_masking(np.corrcoef(ts2[u,:]*ts2[v,:]))
#     brain_features=0.5*(data1+data2)
#     eFC.append(brain_features)
# eFC=np.array(eFC)


# #### Save brain eFC data and also the MAD with only 10 features
# ncomponents=10
# # X=eFC
# # mdic={'X':X}
# # savemat(path_output_data+"X_edges_ALL.mat",mdic)

# X=compute_MAD_and_extract_data_with_exact_number(eFC,features=5)
# mdic={'X':X}
# savemat(path_output_data+f'X_edges_ALL_MAD_{ncomponents}.mat',mdic)




## Loading the data about the violating triangles
input_folder=path_output_data
with open(f'{input_folder}all_data_triangles.pkl', 'rb') as f:
    all_data_triangles=pkl.load(f)
## Loading all data about the homological scaffold
input_folder=path_output_data
with open(f'{input_folder}all_data_scaffold.pkl', 'rb') as f:
    all_data_scaffold=pkl.load(f)

print("Computing triangles...")
np_array_triangles=[]
for key in tqdm.tqdm(sorted(list(set([i[0] for i in all_data_triangles.keys()])))):
    np_array_triangles.append(0.5*(all_data_triangles[(key,'1')]+all_data_triangles[(key,'2')]))
np_array_triangles=np.array(np_array_triangles)


np_array_scaffold=[]
for key in tqdm.tqdm(sorted(list(set([i[0] for i in all_data_scaffold.keys()])))):
    np_array_scaffold.append(0.5*(all_data_scaffold[(key,'1')]+all_data_scaffold[(key,'2')]))
np_array_scaffold=np.array(np_array_scaffold)

#### Save violating triangles data and also the MAD with only 10 features
ncomponents=10
X=np_array_triangles
mdic={'X':X}
savemat(path_output_data+"X_triangles_ALL.mat",mdic)

X=compute_MAD_and_extract_data_with_exact_number(np_array_triangles,features=ncomponents)
mdic={'X':X}
savemat(path_output_data+f"X_triangles_ALL_MAD_{ncomponents}.mat",mdic)

print("Computing scaffold...")
#### Save scaffold data and also the MAD with only 10 features
X=np_array_scaffold
mdic={'X':X}
savemat(path_output_data+"X_scaffold_ALL.mat",mdic)

X=compute_MAD_and_extract_data_with_exact_number(np_array_scaffold,features=ncomponents)
mdic={'X':X}
savemat(path_output_data+f"X_scaffold_ALL_MAD_{ncomponents}.mat",mdic)


###### Do the same as before but on a local level ( 7 Yeo Networks) (within the network)

print("Computing on a local level (Yeo networks)...")


### Associate to each networks, the corresponding indexes with the specific Yeo networks
list_yeo_networks={'VIS':0,'SM':1,'DA':2,'VA':3,'L':4,'FP':5,'DMN':6,'SC':7,'ALL':10}
list_nodes_triplets_indexes={i:0 for i in list_yeo_networks}

for NET_selected in tqdm.tqdm(list_yeo_networks): 
    if NET_selected != 'ALL':
        net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
        nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
        list_Yeonetwork_indexes_triplets=[]
        #find the indices of triplets with ALL the nodes in the current Yeo network
        for idx,l in enumerate(combinations(np.arange(N_nodes),3)):
            i,j,k=l
            if (i in nodes_Yeonetwork) and (j in nodes_Yeonetwork) and (k in nodes_Yeonetwork):
                list_Yeonetwork_indexes_triplets.append(idx)

    else:
        list_Yeonetwork_indexes_triplets=np.arange(N_triangles)

    list_Yeonetwork_indexes_triplets=np.array(list_Yeonetwork_indexes_triplets)
    list_nodes_triplets_indexes[NET_selected]=list_Yeonetwork_indexes_triplets
    
## These indices are for FC and scaffold
list_nodes_edges_indexes={i:0 for i in list_yeo_networks}  
for NET_selected in tqdm.tqdm(list_yeo_networks): 
    if NET_selected != 'ALL':
        net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
        nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
        list_Yeonetwork_indexes_edges=[]
        #find the indices of edges with ALL the nodes in the current Yeo network
        for idx,l in enumerate(combinations(np.arange(N_nodes),2)):
            i,j=l
            if (i in nodes_Yeonetwork) and (j in nodes_Yeonetwork):
                list_Yeonetwork_indexes_edges.append(idx)

    else:
        list_Yeonetwork_indexes_edges=np.arange(N_edges)

    list_Yeonetwork_indexes_edges=np.array(list_Yeonetwork_indexes_edges)
    list_nodes_edges_indexes[NET_selected]=list_Yeonetwork_indexes_edges

### These indices are for eFC    
map_idx_edges={idx:list(val) for idx,val in enumerate(combinations(np.arange(N_nodes),2))}    
list_nodes_eFC_indexes={i:0 for i in list_yeo_networks}  
for NET_selected in tqdm.tqdm(list_yeo_networks): 
    if NET_selected != 'ALL':
        net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
        nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
        list_Yeonetwork_indexes_edges=[]
        #find the indices of edges with at least one node in the current Yeo network
        for idx,l in enumerate(combinations(np.arange(N_edges),2)):
            i,j = map_idx_edges[l[0]]
            m,n = map_idx_edges[l[1]]
            if (i in nodes_Yeonetwork) and (j in nodes_Yeonetwork) and (m in nodes_Yeonetwork) and (n in nodes_Yeonetwork):
                list_Yeonetwork_indexes_edges.append(idx)

    else:
        list_Yeonetwork_indexes_edges=np.arange(len(list(combinations(np.arange(N_edges),2))))

    list_Yeonetwork_indexes_edges=np.array(list_Yeonetwork_indexes_edges)
    list_nodes_eFC_indexes[NET_selected]=list_Yeonetwork_indexes_edges
    

## Compute all the metrics for Nodal, Edges, Triangles, Scaffold

for sel_yeo in tqdm.tqdm(['VIS','SM','DA','VA','L','FP','DMN','SC']):
    print(f"Computing on a local level (Yeo networks) --- {sel_yeo}...")
    ##This is to extract the nodal/FC information
    brain_data=[]
    for path_subjID in tqdm.tqdm(sorted(glob.glob(path_HCP_data+'*'))):
        subjID=path_subjID.strip().split('/')[-1]
        input_folder1=f'{path_HCP_data}{subjID}/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        input_folder2=f'{path_HCP_data}{subjID}/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data1=upper_tri_masking(np.corrcoef(loadmat(input_folder1)['TS']))[list_nodes_edges_indexes[sel_yeo]]
        data2=upper_tri_masking(np.corrcoef(loadmat(input_folder2)['TS']))[list_nodes_edges_indexes[sel_yeo]]
        brain_features=0.5*(data1+data2)
        brain_data.append(brain_features)
    brain_data=np.array(brain_data)
    print(sel_yeo,np.shape(brain_data))

    for ncomponents in [10]:
        try:
            X=compute_MAD_and_extract_data_with_exact_number(brain_data,features=ncomponents)
            mdic={'X':X}
            savemat(path_output_data + f'X_BOLD_{sel_yeo}_MAD_{ncomponents}.mat',mdic)
        except:
            continue
    
    mdic={'X':brain_data}   
    savemat(path_output_data + f'X_BOLD_{sel_yeo}.mat',mdic)

    ##This is to extract the edge/eFC information
    brain_data=[]
    for path_subjID in tqdm.tqdm(sorted(glob.glob(path_HCP_data+'*'))):
        subjID=path_subjID.strip().split('/')[-1]
        input_folder1=f'{path_HCP_data}{subjID}/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        input_folder2=f'{path_HCP_data}{subjID}/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        eTS_data1=compute_eFC(loadmat(input_folder1)['TS'])
        eTS_data2=compute_eFC(loadmat(input_folder2)['TS'])
        data1=upper_tri_masking(eTS_data1)[list_nodes_eFC_indexes[sel_yeo]]
        data2=upper_tri_masking(eTS_data2)[list_nodes_eFC_indexes[sel_yeo]]
        brain_features=0.5*(data1+data2)
        brain_data.append(brain_features)
    brain_data=np.array(brain_data)
    
    #for ncomponents in [5,10,15,30,50,100,150,200,400]:
    for ncomponents in [10]:
        try:
            X=compute_MAD_and_extract_data_with_exact_number(brain_data,features=ncomponents)
            mdic={'X':X}
            savemat(path_output_data + f'X_edges_{sel_yeo}_MAD_{ncomponents}.mat',mdic)
        except:
            continue

    mdic={'X':brain_data}   
    #np.save(f'/allusers/andrea/Dropbox/Postdoc/EPFL/Research_02_brain_project/Rebuttal_analyses/code/Python_analyse_CCA/sCCA-master/sCCA/code/final/X_edges_{sel_yeo}.mat',brain_data)
    savemat(path_output_data + f'X_edges_{sel_yeo}.mat',mdic)

    
    ##This is to extract the triangles information
    np_array_triangles=[]
    for key in tqdm.tqdm(sorted(list(set([i[0] for i in all_data_triangles.keys()])))):
        np_array_triangles.append(0.5*(all_data_triangles[(key,'1')][list_nodes_triplets_indexes[sel_yeo]]+
                                       all_data_triangles[(key,'2')][list_nodes_triplets_indexes[sel_yeo]]))
    np_array_triangles=np.array(np_array_triangles)
    
    for ncomponents in [10]:
        try:
            X=compute_MAD_and_extract_data_with_exact_number(np_array_triangles,features=ncomponents)
            mdic={'X':X}
            savemat(path_output_data + f'X_triangles_{sel_yeo}_MAD_{ncomponents}.mat',mdic)
        except:
            continue

    mdic={'X':np_array_triangles}   
    savemat(path_output_data+ f'X_triangles_{sel_yeo}.mat',mdic)
    
    
    
    ##This is to extract the scaffold information
    np_array_scaffold=[]
    for key in tqdm.tqdm(sorted(list(set([i[0] for i in all_data_scaffold.keys()])))):
        np_array_scaffold.append(0.5*(all_data_scaffold[(key,'1')][list_nodes_edges_indexes[sel_yeo]]+
                                      all_data_scaffold[(key,'2')][list_nodes_edges_indexes[sel_yeo]]))
    np_array_scaffold=np.array(np_array_scaffold)
    
    #for ncomponents in [5,10,15,30,50,100,150,200,400]:
    for ncomponents in [10]:
        try:
            X=compute_MAD_and_extract_data_with_exact_number(np_array_scaffold,features=ncomponents)
            mdic={'X':X}
            savemat(path_output_data + f'X_scaffold_{sel_yeo}_MAD_{ncomponents}.mat',mdic)
        except:
            continue

    mdic={'X':np_array_scaffold}   
    savemat(path_output_data+ f'X_scaffold_{sel_yeo}.mat',mdic)
 
        
    
