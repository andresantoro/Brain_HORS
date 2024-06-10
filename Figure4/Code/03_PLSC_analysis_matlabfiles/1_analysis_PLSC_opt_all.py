from BehavPLS_opt import BehavPLS
# from plot import *
from compute_opt import *
from configparser import ConfigParser
from argparse import ArgumentParser

import numpy as np
import pickle
import typer
from scipy.io import loadmat
import random

app = typer.Typer()
import yaml

@app.command()


def set_default_parameter():
    ##Number of participants
    nb= 80
    # Number of permutations for significance testing
    nPer = 1000
    # Number of bootstrap iterations
    nBoot = 1000
    # Seed
    seed = 10
    # signficant level for statistical testing
    seuil=0.05

    return(nb,nPer,nBoot,seed,seuil)

def save_pkl (data, name): 
    with open(f'./pkl/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'./pkl/{name}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)




if __name__ == "__main__":
    path_output_data="./../../Results/brain_features/"
    path_output_PLSCresults="./../../Results/PLSC_results_numpy/"
 
    ### Load data bootstrap IDs
    list_boostrap_id=[]
    with open('BOLD_bootstrap_sample_ID.txt','r+') as f:
        for l in f:
            list_boostrap_id.append(list(map(int,l.strip()[1:-1].split(','))))
    list_samples=np.array(list_boostrap_id)
    # for type_ in ['MAD']:
    for net in ['VIS','SM','DA','VA','L','DMN','FP','SC']:
        for method in ['BOLD','edges','scaffold','triangles']:
            #for sss in [800,1500]: #[5,10,15,30,50,100,150,200,400]: 
            # for sss in [5,10]:
                Y= loadmat('matrix_HCP_data.mat')
                Ylabel=[i[0] for i in Y['domains'][0]]
                Y=Y['Bpca']
                X=loadmat(path_output_data+f'X_{method}_{net}.mat')['X']
                X=pd.DataFrame(np.array(X)) ### This is a "brain" matrix of size (Nsubjects, Nbrainfeatures
                Y=pd.DataFrame(np.array(Y)) ### This is a "behavioral" matrix of size (Nsubjects, Nbehavioral_scores)
        
                nb,nPer,nBoot,seed,seuil = set_default_parameter()
                #type_='MAD'
        
        
                covariance_all=[]
                correlation_all=[]
                for i in range(100):
                    indexes=list_samples[i]
                    dataset=BehavPLS(X.iloc[indexes,:],Y.iloc[indexes,:],nb_sub=nb,nPerms=nPer,nBoot=nBoot,seed=seed,seuil=seuil,verbose=True)
        
                    res_decompo = dataset.run_decomposition()
                    # save_pkl(res_decompo, f"pls_res_{type_}")
                   
                   
                    res_permu=dataset.permutation()
                    # res_bootstrap = dataset.bootstrap()
                    
                    #print(res_permu['P_val'][res_permu['sig_LC']], res_permu['sig_LC'],varexp(res_decompo['S'])[res_permu['sig_LC']])
                    varexplained_all=varexp(res_decompo['S'])[res_permu['sig_LC']]
                    if len(varexplained_all)==0:
                        covariance_all.append([i,0,0])
                        correlation_all.append([i,0,0])
                    else:
                        covariance_all.append([i,varexplained_all[0],np.sum(varexplained_all)])
                        for kkk in res_permu['sig_LC']:
                            current_correlation=np.corrcoef(np.array(np.matmul(np.array(X.iloc[indexes,:]),res_decompo['V']))[:,kkk],
                                                        np.array(np.matmul(np.array(Y.iloc[indexes,:]),res_decompo['U']))[:,kkk])
                        
                            correlation_all.append([i,current_correlation[0,1],kkk])
                        # res_bootstrap = dataset.bootstrap()
                        # save_pkl(res_bootstrap, f"boot_res_{type_}_{sss}")
                covariance_all=np.array(covariance_all)
                correlation_all=np.array(correlation_all)
                
                with open(path_output_PLSCresults + f'PLSC_{net}_{method}_covariance.npy', 'wb') as f:
                    np.save(f, covariance_all)
                with open(path_output_PLSCresults + f'PLSC  _{net}_{method}_correlation.npy', 'wb') as f:
                    np.save(f, correlation_all)
                    

        




