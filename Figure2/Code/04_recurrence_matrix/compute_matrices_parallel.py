import numpy as np
import matplotlib.pyplot as plt
import tqdm
import glob
import scipy.io as sio
import h5py
import networkx as nx
from itertools import combinations
from scipy.stats import pearsonr,zscore
import copy
from scipy.io import loadmat


if __name__ == "__main__": 
    orientation='RL'
    path_HCP_data='/home/asantoro/HCP/'
    path_results_HO='./../../Results/HO_localmeasures/'
    path_output='./../../Results/HO_recurrence_plot/'
    N=119

    u,v=np.triu_indices(N,1,N) ## Taking the upper triangular indices for the edge time series
    subjID_all=sorted(glob.glob(path_HCP_data+'*'))[0:100]
    for idx_subj,subjID_path in enumerate(subjID_all):
        subjID=subjID_path.split('/')[-1]
        print(subjID)
        
        ##Load BOLD time series
        data_BOLD=zscore(loadmat(f'{path_HCP_data}{subjID}/TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_{orientation}.mat')['TS'],axis=1) 
        
        ##Compute edge time series
        data_edges=data_BOLD[u,:]*data_BOLD[v,:]
        
        ##Load Triangle time series
        data=h5py.File(f'{path_results_HO}HO_triangles_{subjID}_{orientation}.hd5','r+')
        T=len(data)
        HORS_all_triangles=np.zeros((int(N*(N-1)*(N-2)/6),T))
        for t in tqdm.tqdm(range(T)):
            HORS_all_triangles[:,t]=copy.deepcopy(data[str(t+1)][:])
        data.close()

        ##Load Scaffold time series 
        data_scaffold_persistence=h5py.File(f'{path_results_HO}HO_scaffold_persistence_{subjID}_{orientation}.hd5','r+')
        data_scaffold_frequency=h5py.File(f'{path_results_HO}HO_scaffold_{subjID}_{orientation}.hd5','r+')
        HORS_all_scaffold=np.zeros((int(N*(N-1)/2),T))
        HORS_all_scaffold_frequency=np.zeros((int(N*(N-1)/2),T))
        HORS_all_scaffold_persistence = np.zeros((int(N*(N-1)/2),T))
        for t in tqdm.tqdm(range(T)):
            HORS_all_scaffold[:,t]=copy.deepcopy(np.nan_to_num( (data_scaffold_persistence[str(t+1)][:][u,v]) / (data_scaffold_frequency[str(t+1)][:][u,v]),posinf=0))
            HORS_all_scaffold_frequency[:,t]=copy.deepcopy(data_scaffold_frequency[str(t+1)][:][u,v])
            HORS_all_scaffold_persistence[:,t]=copy.deepcopy(data_scaffold_persistence[str(t+1)][:][u,v])
        data_scaffold_persistence.close()
        data_scaffold_frequency.close()

        
        current_similarity_BOLD=np.corrcoef(data_BOLD.T)
        current_similarity_edges=np.corrcoef(data_edges.T)
        current_similarity_triangles=np.corrcoef(HORS_all_triangles.T)
        current_similarity_scaffold=np.corrcoef(HORS_all_scaffold.T)
        current_similarity_scaffold_p=np.corrcoef(HORS_all_scaffold_persistence.T)
        current_similarity_scaffold_f=np.corrcoef(HORS_all_scaffold_frequency.T)
        if idx_subj==0:
            similarity_BOLD=current_similarity_BOLD
            similarity_edges=current_similarity_edges
            similarity_triangles=current_similarity_triangles
            similarity_scaffold=current_similarity_scaffold
            similarity_scaffold_p=current_similarity_scaffold_p
            similarity_scaffold_f=current_similarity_scaffold_f
        else:
            similarity_BOLD+=current_similarity_BOLD
            similarity_edges+=current_similarity_edges
            similarity_triangles+=current_similarity_triangles
            similarity_scaffold+=current_similarity_scaffold
            similarity_scaffold_p+=current_similarity_scaffold_p
            similarity_scaffold_f+=current_similarity_scaffold_f

        np.save(f'{path_output}similarity_BOLD_0_100_zscored_{orientation}.npy',similarity_BOLD)
        np.save(f'{path_output}similarity_edges_0_100_zscored_{orientation}.npy',similarity_edges)
        np.save(f'{path_output}similarity_triangles_0_100_zscored_{orientation}.npy',similarity_triangles)
        np.save(f'{path_output}similarity_scaffold_0_100_zscored_{orientation}.npy',similarity_scaffold)
        np.save(f'{path_output}similarity_scaffold_0_100_zscored_{orientation}_persistence.npy',similarity_scaffold_p)
        np.save(f'{path_output}similarity_scaffold_0_100_zscored_{orientation}_frequency.npy',similarity_scaffold_f)