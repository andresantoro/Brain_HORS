import numpy as np
from BehavPLS_opt import BehavPLS
from compute_opt import *
import os
import sys
import pickle as pk
from scipy.io import loadmat
import glob
from itertools import combinations


def set_default_parameter():
    ##Number of participants
    nb= 100
    # Number of permutations for significance testing
    nPer = 1000
    # Number of bootstrap iterations
    nBoot = 1000
    # Seed
    seed = 10
    # signficant level for statistical testing
    seuil=0.05
    return(nb,nPer,nBoot,seed,seuil)  

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]  


def compute_eFC(data_TS):
    '''Compute the Edge Functional connectivty
    starting from a BOLD time series'''
    N,T=np.shape(data_TS)
    u,v=np.triu_indices(N,k=1,m=N)
    data_eFC=data_TS[u,:]*data_TS[v,:]
    return(np.corrcoef(data_eFC))


def pairwise_correlation(A, B):
    '''
    Compute the Pearson correlation coefficient between two matrices

    input: A -> numpy 2D array
           B -> numpy 2D array

    output: matrix encoding all the Pearson correlation'''

    
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm /  (np.sqrt(
        np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))

def compute_edgeTS(time_series):
    N, T = time_series.shape
    u, v = np.triu_indices(n=N, k=1,m=N)
    return(np.array(time_series[u,:]*time_series[v,:]))

def save_pkl (data, name,path_output='./pkl/'): 
    with open(f'{path_output}{name}.pkl', 'wb') as f:
        pk.dump(data, f)
#    with open(f'./pkl/{name}.pkl', 'rb') as f:
#        loaded_dict = pk.load(f)

def load_Xdata(method_label,path_subjects,input_folder_path='./../../results/HOI_info/rest/'):
    if method_label == 'BOLD':
        X_mat= load_BOLD_data(path_subjects)
    if method_label == 'edges':
        X_mat= load_edges_data(path_subjects)
    if method_label == 'triangles':
        X_mat= load_triangles_data(path_subjects)
    if method_label == 'scaffold':
        X_mat= load_scaffold_data(path_subjects)
    if method_label == 'Oinfo':
        X_mat= load_infotheory_data(input_folder=input_folder_path, measure=method_label)
    if method_label == 'Sinfo':
        X_mat= load_infotheory_data(input_folder=input_folder_path, measure=method_label)    
    if method_label == 'DTC':
        X_mat= load_infotheory_data(input_folder=input_folder_path, measure=method_label)
    return (X_mat)

def loading_yeo(path="./../../../Misc/yeo_RS7_Schaefer100S.mat"):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path)['yeoROIs']])
    yeoROI_dict={label_Yeo:np.where(yeoROIs==idx_Yeo)[0] for idx_Yeo,label_Yeo in enumerate(['VIS','SM','DA','VA','L','FP','DMN','SC','C'])}
    yeoROI_dict['SC']=np.array(sorted(np.hstack((yeoROI_dict['SC'],yeoROI_dict['C']))))
    del yeoROI_dict['C']
    return(yeoROI_dict)


def load_Xdata_subnetworks(method_label,label_YEO,path_subjects,yeo_map,input_folder_path='./../../results/HOI_info/rest/'):
    map_FC,map_eFC,map_triangles=yeo_map
    if method_label == 'BOLD':
        X_mat= load_BOLD_data_subnetwork(path_subjects,map_FC,label_YEO)
    if method_label == 'edges':
        X_mat= load_edges_data_subnetwork(path_subjects,map_eFC,label_YEO)
    if method_label == 'triangles':
        X_mat= load_triangles_data_subnetwork(path_subjects,map_triangles,label_YEO)
    if method_label == 'scaffold':
        X_mat= load_scaffold_data_subnetwork(path_subjects,map_FC,label_YEO)
    if method_label == 'Oinfo':
        X_mat = load_infotheory_data_subnetwork(input_folder_path,map_triangles, label_YEO,measure=method_label)
    if method_label == 'Sinfo':
        X_mat = load_infotheory_data_subnetwork(input_folder_path,map_triangles, label_YEO,measure=method_label)
    if method_label == 'DTC':
        X_mat = load_infotheory_data_subnetwork(input_folder_path,map_triangles, label_YEO,measure=method_label)
    return (X_mat)




def load_BOLD_data(path_subjects):
    subject_FC_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    # print(glob.glob(path_subjects+'*'))
    for idx,i in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+i+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                

        ##Loading day 2
        path_data=path_subjects+i+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_FC_data[(i,1)]=upper_tri_masking(np.corrcoef(data_day1))
        subject_FC_data[(i,2)]=upper_tri_masking(np.corrcoef(data_day2))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_FC_data[(subjID,1)]+subject_FC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_BOLD_data_subnetwork(path_subjects,yeoROI_dict,label_YEO):
    # if isinstance(label_YEO, str):
    #     indices_yeo=yeoROI_dict[label_YEO]
    # else:
    #     indices_yeo=[]
    #     for l in label_YEO:
    #         indices_yeo.extend(yeoROI_dict[l])
    #     indices_yeo=np.array(sorted(list(set(indices_yeo))))
    indices_yeo=yeoROI_dict[label_YEO]
    subject_FC_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,i in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+i+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                

        ##Loading day 2
        path_data=path_subjects+i+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_FC_data[(i,1)]=upper_tri_masking(np.corrcoef(data_day1))[indices_yeo]
        subject_FC_data[(i,2)]=upper_tri_masking(np.corrcoef(data_day2))[indices_yeo]
        # subject_FC_data[(i,1)]=upper_tri_masking(np.corrcoef(data_day1[:,indices_yeo]))
        # subject_FC_data[(i,2)]=upper_tri_masking(np.corrcoef(data_day2[:,indices_yeo]))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_FC_data[(subjID,1)]+subject_FC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_edges_data(path_subjects):
    subject_eFC_data={}

    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,subjID in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+subjID+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                
        
        ##Loading day 2
        path_data=path_subjects+subjID+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        edges1=compute_edgeTS(data_day1)
        edges2=compute_edgeTS(data_day2)
        subject_eFC_data[(subjID,1)]=upper_tri_masking(pairwise_correlation(edges1.T,edges1.T))
        subject_eFC_data[(subjID,2)]=upper_tri_masking(pairwise_correlation(edges2.T,edges2.T))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_eFC_data[(subjID,1)]+subject_eFC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_edges_data_subnetwork(path_subjects,yeoROI_dict,label_YEO):
    # if isinstance(label_YEO, str):
    #     indices_yeo=yeoROI_dict[label_YEO]
    # else:
    #     indices_yeo=[]
    #     for l in label_YEO:
    #         indices_yeo.extend(yeoROI_dict[l])
    #     indices_yeo=np.array(sorted(list(set(indices_yeo))))
    indices_yeo=yeoROI_dict[label_YEO]
    subject_eFC_data={}

    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    for idx,subjID in enumerate(list_subjs):
        ##Loading day 1
        path_data=path_subjects+subjID+'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day1=loadmat(path_data)['TS'] ## Size: (119, 1200)                
        
        ##Loading day 2
        path_data=path_subjects+subjID+'/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat'
        data_day2=loadmat(path_data)['TS'] ## Size: (119, 1200)
        
        subject_eFC_data[(subjID,1)]=upper_tri_masking(compute_eFC(data_day1))[indices_yeo]
        subject_eFC_data[(subjID,2)]=upper_tri_masking(compute_eFC(data_day2))[indices_yeo]
        # subject_eFC_data[(subjID,1)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day1[indices_yeo,:])))
        # subject_eFC_data[(subjID,2)]=upper_tri_masking(np.corrcoef(compute_edgeTS(data_day2[indices_yeo,:])))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_eFC_data[(subjID,1)]+subject_eFC_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_triangles_data(path_subjects,N=119,T=1200):
    subject_triangles_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    if os.path.isfile('data_triangles_REST.pkl'):
        subject_triangles_data={}
        with open('data_triangles_REST.pkl','rb+') as f:
            subject_triangles_data=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_triangles_data[(subjID,1)]+subject_triangles_data[(subjID,2)]))
    X_mat=np.array(X_mat)
    return(X_mat)

def load_triangles_data_subnetwork(path_subjects,yeoROI_dict,label_YEO,path_data_triangles='./../../Results/brain_features/'):
    indices_yeo=yeoROI_dict[label_YEO]

    # indices_yeo_all=[]
    # if isinstance(label_YEO, str):
    #     indices_yeo=yeoROI_dict[label_YEO]
    # else:
    #     indices_yeo=[]
    #     for l in label_YEO:
    #         indices_yeo.extend(yeoROI_dict[l])
    #     indices_yeo=np.array(sorted(list(set(indices_yeo))))
    # for idx_triangles,(i,j,k) in enumerate(combinations(np.arange(N),3)):
    #         flag=[i in indices_yeo, j in indices_yeo, k in indices_yeo]
    #         if sum(flag) == flag_triangles: ## All the nodes belong to the same Yeo networks
    #             indices_yeo_all.append(idx_triangles)
    # indices_yeo_all=np.array(indices_yeo_all)

    subject_triangles_data={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    path=path_data_triangles
    if os.path.isfile(path+'all_data_triangles.pkl'):
        subject_triangles_data={}
        with open(path+'all_data_triangles.pkl','rb+') as f:
            subject_triangles_data=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(subject_triangles_data[(subjID,'1')][indices_yeo]+subject_triangles_data[(subjID,'2')][indices_yeo]))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_scaffold_data(path_subjects,N=119,T=1200,path_data='./../../Results/brain_features/'):
    data_scaffold_REST={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    if os.path.isfile(path_data+'data_scaffold_REST.pkl'):
        data_scaffold_REST={}
        with open(path_data+'data_scaffold_REST.pkl','rb+') as f:
            data_scaffold_REST=pk.load(f)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(upper_tri_masking(data_scaffold_REST[(subjID,1)])+upper_tri_masking(data_scaffold_REST[(subjID,2)])))
    X_mat=np.array(X_mat)
    return(X_mat)


def load_scaffold_data_subnetwork(path_subjects,yeoROI_dict,label_YEO,path_data_scaffold='./../../Results/brain_features/'):
    # if isinstance(label_YEO, str):
    #     indices_yeo=yeoROI_dict[label_YEO]
    # else:
    #     indices_yeo=[]
    #     for l in label_YEO:
    #         indices_yeo.extend(yeoROI_dict[l])
    #     indices_yeo=np.array(sorted(list(set(indices_yeo))))
    indices_yeo=yeoROI_dict[label_YEO]
    data_scaffold_REST={}
    list_subjs=np.array(sorted([i.split('/')[-1] for i in glob.glob(path_subjects+'*')]))
    #print(list_subjs)
    path=path_data_scaffold
    if os.path.isfile(path+'all_data_scaffold.pkl'):
        data_scaffold_REST={}
        with open(path+'all_data_scaffold.pkl','rb+') as f:
            data_scaffold_REST=pk.load(f)
    #print(data_scaffold_REST)
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(data_scaffold_REST[(subjID,'1')][indices_yeo]+ data_scaffold_REST[(subjID,'2')][indices_yeo]))
    X_mat=np.array(X_mat)
    return(X_mat)

### Loading the data about triplets with O-info, S-info and DTC
def load_infotheory_data(input_folder='./../../results/HOI_info/rest/', measure='Oinfo'):    
    all_data={}
    list_subjs=[]
    for filename in sorted(glob.glob(input_folder+'*')):
        ID,day,orientation=filename.strip().split('/')[-1].split('.')[0].split('_')[3:6]
        list_subjs.append(ID)
    
        current_data=loadmat(filename)['HOI_info']##Load the data
        cleaned_data=dict(zip(list(current_data.dtype.fields.keys()),list(current_data[0][0])))
        if orientation == 'LR':
            all_data[(ID,day)]=cleaned_data
    list_subjs=list(set(list_subjs))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(np.ravel(all_data[(subjID,'1')][measure])+np.ravel(all_data[(subjID,'2')][measure])))
    X_mat=np.array(X_mat)
    return(X_mat)

### Loading the data about triplets with O-info, S-info and DTC for the subnetworks
def load_infotheory_data_subnetwork(input_folder,yeoROI_dict,label_YEO,flag_triplets=3,N=119,measure='Oinfo'):
    # indices_yeo_all=[]
    # if isinstance(label_YEO, str):
    #     indices_yeo=yeoROI_dict[label_YEO]
    # else:
    #     indices_yeo=[]
    #     for l in label_YEO:
    #         indices_yeo.extend(yeoROI_dict[l])
    #     indices_yeo=np.array(sorted(list(set(indices_yeo))))
    # for idx_triangles,(i,j,k) in enumerate(combinations(np.arange(N),3)):
    #         flag=[i in indices_yeo, j in indices_yeo, k in indices_yeo]
    #         if sum(flag) == flag_triplets: ## All the nodes belong to the same Yeo networks
    #             indices_yeo_all.append(idx_triangles)
    # indices_yeo_all=np.array(indices_yeo_all)
    indices_yeo=yeoROI_dict[label_YEO]
    
    all_data={}
    list_subjs=[]
    for filename in sorted(glob.glob(input_folder+'*')):
        ID,day,orientation=filename.strip().split('/')[-1].split('.')[0].split('_')[3:6]
        list_subjs.append(ID)
        
        current_data=loadmat(filename)['HOI_info']##Load the data
        cleaned_data=dict(zip(list(current_data.dtype.fields.keys()),list(current_data[0][0])))
        if orientation == 'LR':
            all_data[(ID,day)]=cleaned_data
    list_subjs=sorted(list(set(list_subjs)))
    X_mat=[]
    for subjID in list_subjs:
        X_mat.append(0.5*(np.ravel(all_data[(subjID,'1')][measure][indices_yeo])+np.ravel(all_data[(subjID,'2')][measure][indices_yeo])))
    X_mat=np.array(X_mat)
    return(X_mat)



def extract_indices_localyeo(N_nodes=119,path_rsn="./../../../Misc/yeo_RS7_Schaefer100S.mat"):
    ##Loading the yeoROIS
    yeoROIs=np.array([i[0]-1 for i in loadmat(path_rsn)['yeoROIs']])
    N_edges=int((N_nodes)*(N_nodes-1)/2)
    N_triangles=int((N_nodes)*(N_nodes-1)*(N_nodes-2)/6)
    ### Associate to each networks, the corresponding indexes with the specific Yeo networks
    list_yeo_networks={'VIS':0,'SM':1,'DA':2,'VA':3,'L':4,'FP':5,'DMN':6,'SC':7,'ALL':10}
    list_nodes_triplets_indexes={i:0 for i in list_yeo_networks}

    for NET_selected in list_yeo_networks: 
        if NET_selected != 'ALL':
            net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
            nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
            list_Yeonetwork_indexes_triplets=[]
            #find the indices of triplets with ALL the nodes in the current Yeo network
            for idx,(i,j,k) in enumerate(combinations(np.arange(N_nodes),3)):
                flag=[i in nodes_Yeonetwork, j in nodes_Yeonetwork, k in nodes_Yeonetwork]
                if sum(flag) == 3: ## All the nodes belong to the same Yeo networks
                    list_Yeonetwork_indexes_triplets.append(idx)
    
        else:
            list_Yeonetwork_indexes_triplets=np.arange(N_triangles)
    
        list_Yeonetwork_indexes_triplets=np.array(list_Yeonetwork_indexes_triplets)
        list_nodes_triplets_indexes[NET_selected]=list_Yeonetwork_indexes_triplets
    
    ## These indices are for FC and scaffold
    list_nodes_edges_indexes={i:0 for i in list_yeo_networks}  
    for NET_selected in list_yeo_networks: 
        if NET_selected != 'ALL':
            net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
            nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
            list_Yeonetwork_indexes_edges=[]
            #find the indices of edges with ALL the nodes in the current Yeo network
            for idx,(i,j) in enumerate(combinations(np.arange(N_nodes),2)):
                flag=[i in nodes_Yeonetwork, j in nodes_Yeonetwork]
                if sum(flag) == 2: ## All the nodes belong to the same Yeo networks
                    list_Yeonetwork_indexes_edges.append(idx)
    
        else:
            list_Yeonetwork_indexes_edges=np.arange(N_edges)
    
        list_Yeonetwork_indexes_edges=np.array(list_Yeonetwork_indexes_edges)
        list_nodes_edges_indexes[NET_selected]=list_Yeonetwork_indexes_edges

    ### These indices are for eFC    
    map_idx_edges={idx:list(val) for idx,val in enumerate(combinations(np.arange(N_nodes),2))}    
    list_nodes_eFC_indexes={i:0 for i in list_yeo_networks}  
    for NET_selected in list_yeo_networks: 
        if NET_selected != 'ALL':
            net_sel=list_yeo_networks[NET_selected] ## This is current Yeo network
            nodes_Yeonetwork=frozenset(np.where(yeoROIs==net_sel)[0])
            list_Yeonetwork_indexes_edges=[]
            #find the indices of edges with at least one node in the current Yeo network
            for idx,l in enumerate(combinations(np.arange(N_edges),2)):
                i,j = map_idx_edges[l[0]]
                m,n = map_idx_edges[l[1]]
                flag=[i in nodes_Yeonetwork, j in nodes_Yeonetwork, m in nodes_Yeonetwork, n in nodes_Yeonetwork]
                if sum(flag) == 4:
                    list_Yeonetwork_indexes_edges.append(idx)
    
        else:
            list_Yeonetwork_indexes_edges=np.arange(len(list(combinations(np.arange(N_edges),2))))
    
        list_Yeonetwork_indexes_edges=np.array(list_Yeonetwork_indexes_edges)
        list_nodes_eFC_indexes[NET_selected]=list_Yeonetwork_indexes_edges
    
    return (list_nodes_edges_indexes,list_nodes_eFC_indexes,list_nodes_triplets_indexes)



            

if __name__ == "__main__":
    MIN_BOOT=0
    MAX_BOOT=100
    total_number_subjects=100
    max_subjects_bootstrap=80
    
    path_data_behavioral='matrix_HCP_data.mat'
    Y=loadmat(path_data_behavioral)
    Ylabel=[i[0] for i in Y['domains'][0]] ## This corresponds to the behavioral data labels (i.e. 10 cognitive scores)
    Y=Y['Bpca'] ## (these are the scores for the different subjects, array of 100x10)
    nb,nPer,nBoot,seed,seuil = set_default_parameter()
    flag_networks=0
    
    output_path='./../../Results/pkl/'
    HCP_datapath='/home/andrea/miplabsrv2/HCP/'
    
    list_boostrap_id=[]
    ### Load data bootstrap IDs
    with open('BOLD_bootstrap_sample_ID.txt','r+') as f:
        for l in f:
            list_boostrap_id.append(list(map(int,l.strip()[1:-1].split(','))))
    list_boostrap_id=np.array(list_boostrap_id)
    #list_methods=['Oinfo','Sinfo','DTC']
    #list_methods=['BOLD']
    list_methods=['Oinfo','Sinfo','DTC']
    # list_methods=['edges']
    
    if flag_networks == 0:
        functional_network='ALL'
        for method in list_methods:
             nb=max_subjects_bootstrap
             X=load_Xdata(method,path_subjects=HCP_datapath,input_folder_path="./../../../Supplementary/HO_Gatica/Results/")
             for id_bootstrap in range(MIN_BOOT,MAX_BOOT):
                sample_ID=np.array(list_boostrap_id[id_bootstrap])
                X_current=X[sample_ID,:]
                Y_current=Y[sample_ID,:]
                print("doing the following method: %s --- Functional network: %s -- Round: %d" % (method,functional_network,id_bootstrap))
                dataset=BehavPLS(X_current,Y_current,nb_sub=nb,nPerms=nPer,nBoot=nBoot,seed=seed,seuil=seuil,verbose=True)
                res_decompo = dataset.run_decomposition()
                save_pkl(res_decompo, f"pls_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)
               
                res_permu=dataset.permutation()
                save_pkl(res_permu, f"perm_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)
                    
                # res_bootstrap = dataset.bootstrap()
                # save_pkl(res_bootstrap, f"boot_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)
                print(sample_ID, file=sys.stderr)

    else:
        ## The output of this function is a tuple containing the mapping of the indices
        ## of the different order for selecting only specific subnetworks: map_FC,map_eFC,map_triangles
        yeo_map= extract_indices_localyeo(path_rsn="./../../../Misc/yeo_RS7_Schaefer100S.mat")
        
        for method in list_methods:
            list_functional_networks=['VIS','SM','DA','VA','L','FP','DMN','SC']
            for idx1,functional_network in enumerate(list_functional_networks):
                X=load_Xdata_subnetworks(method,functional_network,path_subjects=HCP_datapath,yeo_map=yeo_map,input_folder_path="./../../../Supplementary/HO_Gatica/Results/")
                for id_bootstrap in range(MIN_BOOT,MAX_BOOT):
                    sample_ID=np.array(list_boostrap_id[id_bootstrap])
                    X_current=X[sample_ID,:]
                    Y_current=Y[sample_ID,:]
                    # print(X_current)
                    print("doing the following method: %s --- Functional network: %s -- Round: %d" % (method,functional_network,id_bootstrap))

                    dataset=BehavPLS(X_current,Y_current,nb_sub=nb,nPerms=nPer,nBoot=nBoot,seed=seed,seuil=seuil,verbose=True)
                    res_decompo = dataset.run_decomposition()
                    save_pkl(res_decompo, f"pls_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)
                   
                    res_permu=dataset.permutation()
                    save_pkl(res_permu, f"perm_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)
                        
                    res_bootstrap = dataset.bootstrap()
                    save_pkl(res_bootstrap, f"boot_res_{method}_{functional_network}_iter{id_bootstrap}",path_output=output_path)

                    print(sample_ID, file=sys.stderr)
