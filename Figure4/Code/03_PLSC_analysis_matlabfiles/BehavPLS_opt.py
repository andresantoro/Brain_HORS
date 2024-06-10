from compute_opt import *
import nibabel as nib
import glob
import os
from nilearn.image import clean_img
from nilearn.masking import compute_brain_mask, apply_mask
from collections import defaultdict


class BehavPLS(): 

	def __init__(self, X, Y, nb_sub, nPerms= 100, nBoot=100, seed=1, seuil=0.05,verbose=True,  **kwargs): 
		
		self.brain_data= pd.DataFrame(np.array(X))
		self.behav_data= pd.DataFrame(np.array(Y))
		self.nPerms = nPerms
		self.nsub=nb_sub
		self.nBoot=nBoot
		self.seed=seed
		self.seuil=seuil

	def run_decomposition(self, **kwargs):
		"""
		Standardization and SVD of the covariance matrix between brain_data/X and behav_data/Y
		"""
											
											
		res={}
		print("... Normalisation ...")
		###Z-scoring the data
		self.X_std, self.Y_std = standardization(self.brain_data, self.behav_data)
		res['X']=self.brain_data
		res['Y']=self.behav_data 
		res['X_std']= self.X_std
		res['Y_std']= self.Y_std
	 
		print("...SVD ...")
		self.R=R_cov(self.X_std, self.Y_std)
		self.U,self.S, self.V = SVD(self.R, ICA=True)
		self.ExplainedVarLC =varexp(self.S)
		self.Lx, self.Ly= PLS_scores(self.X_std, self.Y_std, self.U, self.V)

		res['R']=self.R
		res['U']=self.U
		res['S']=self.S
		res['V']=self.V
	   
		return res

	def permutation(self, **kwargs):
		print("...Permutation...")
		res={}
		res['Sp_vect']=permu(self.X_std, self.Y_std, self.U, self.nPerms, self.seed)
		res['P_val'], res['sig_LC'] = myPLS_get_LC_pvals(res['Sp_vect'],self.S,self.nPerms, self.seuil)
		return res


	def bootstrap(self, **kwargs): 
		print("... Bootstrap...")
		res={}
		res= myPLS_bootstrapping(self.brain_data,self.behav_data , self.U,self.V, 
										  self.nBoot,  self.seed)
	   
		return res

