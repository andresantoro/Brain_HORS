from compute_opt import*
import nibabel as nib
import os
import nilearn
import nibabel as nib
from nilearn import plotting, image
import numpy as np
import matplotlib.pyplot as plt
from nilearn.masking import compute_brain_mask, apply_mask

def exp_var(S, Sp_vect, LC_pvals, name): 
    """
    Plot the cumulative explained variance and save it
    
    Inputs 
    -------
    S: ndarray
        Array of singular values
        
    Sp_vect: ndarray
        Array of LC vectors
        
    LC_pvals: ndarray
        Array of LC p-values
        
    name: str
        Name of the plot file to save
        
   
    """
    fig, ax = plt.subplots(figsize=(6, 3),dpi=150)
    
    indices_significant=np.where(LC_pvals<0.05)[0]
    
    # Number of LCs
    nc = np.arange(len(LC_pvals)) + 1
    
    # Create another axes object for secondary y-Axis
    # ax2 = ax.twinx()
    
    # Plot singular values
#     ax.plot(nc, np.diag(S), color='grey', marker='o', fillstyle='none',clip_on=False)
    ax.plot(nc, varexp(S), color='grey', marker='o', fillstyle='none',clip_on=False)
    ax.plot(nc[indices_significant], varexp(S)[indices_significant], color='grey', marker='o',clip_on=False)
    
    # Plot the cumulative explained covariance
    # ax2.plot(nc, (np.cumsum(varexp(S))*100), color='steelblue', ls='--',clip_on=False)
    
    # Labeling & Axes limits
    labels = [f"LC {idx+1} (p={np.round(i, 4)})" for idx, i in enumerate(LC_pvals)]
    #plt.title('Explained Covariance by each LC')
    ax.set_ylabel('Explained Covariance')
    
#     ticks_loc = ax.get_xticks().tolist()
#     ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#     print(len(ticks_loc))
#     ax.set_xticks(ticks_loc)
    ax.set_xticks(np.arange(1, len(LC_pvals)+0.1))
    ax.set_xticklabels(labels, rotation=45,ha='right')
    ax.set_xlim([1, len(LC_pvals)])
    # ax2.set_xticks(nc, labels)
    # ax2.set_ylabel('Explained correlation', color='steelblue')
    # ax2.set_ylim([0, 100])
#     plt.grid()

    # Defining display layout
    plt.tight_layout()
    
    # Save the Plot
    # plt.savefig(f"../Plots/Var/{name}.png", bbox_inches='tight',transparent = True )

    
def print_var(S) :
    """
    Print the explained variance for each latent component (LC).

    Inputs:
    -------
    S: ndarray
        Array of singular values representing the strength of each latent component.
    """
    
    for idx, s in enumerate(varexp(S)):
        print(f"Explained variance for LC{idx+1}: {s}")

    
def save_fMRI(data, mask, title, shape=[91, 109, 91]):
    """
    Convert and save the received data into a Nifti file
    
    Inputs:
    -------
    data: np.array
            input data
    mask: nib.Nifti1Image
            the binary mask
    title: str
            the title of the saved file
    shape: list
            shape of the Nifti file (default=[91, 109, 91])
    """
    # Create an array of the size of the mask
    new_brain = np.zeros(np.shape(mask.get_fdata() > 0))
    
    # Prune voxels to 0 if outside the mask
    new_brain[mask.get_fdata() > 0] = data
    new_brain = new_brain.reshape(shape)
    
    # Convert to Nifti format and save
    nift_fMRI = nib.Nifti1Image(new_brain, mask.affine)
    nib.save(nift_fMRI, f"../Nifti/{title}.nii.gz")
    
def plot_z_slices(img, n_rows, n_cols, title=None, output_file=None):
    """
    Plot axial slices of a 3D NIfTI image.
    
    Inputs:
    -------
    img : nibabel.nifti1.Nifti1Image
        Input 3D NIfTI image.
    n_rows : int
        Number of rows for the plot.
    n_cols : int
        Number of columns for the plot.
    title : str, optional
        Title of the plot.
    output_file : str, optional
        Name of the output file to save the plot.
    """
    all_coords = plotting.find_cut_slices(img, direction="z", n_cuts=n_rows * n_cols)
    ax_size = 3.0
    margin = 0.05
    fig, all_axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * ax_size, n_cols * ax_size),
                                 gridspec_kw={"hspace": margin, "wspace": margin})
    left_right = True
    for coord, ax in zip(all_coords, all_axes.ravel()):
        display = plotting.plot_stat_map(img, cut_coords=[coord], display_mode="z", axes=ax, annotate=False)
        display.annotate(left_right=left_right)
        left_right = False
    if title:
        plt.suptitle(title, fontsize=15)
    if output_file:
        plt.savefig(output_file)
        
def brain_plot_slices(LC_indexes, name): 
    """
    Plot and save brain saliences (z_slices)
    """
    for i in LC_indexes: 
        file = f"../Nifti/LV{i+1}_{name}.nii.gz"
        pic = image.load_img(file)
        plot_z_slices(pic, 5, 5, f"Brain Z-slices, LV: {i+1}", f"../Plots/Brain/Brain Z-slices,LV:{i+1}_{name}")

        
def brain_plot(LC_indexes, name): 
    """
    Plot and Save Brain Saliences (3 slices)
    
    Inputs
    -------
    LC_indexes : list
        List of tuples with the indexes of the LVs to plot
    name : str
        Name of the saliency map
    color : str
        Name of the color for the plot
    
    """
    for i in LC_indexes:
        file = f'../Nifti/LV{i+1}_{name}.nii.gz'
        pic = image.load_img(file)
        plotting.plot_stat_map(pic,
                               display_mode='ortho',
                               draw_cross=False, 
                               annotate=True,
                               black_bg=False)
        plt.savefig(f'../Plots/Brain/LV{i+1}_{name}.png', transparent = True)
    
        
def modify_color(c, index, col ) : 
    """
    Modify a color array with given indexes.

    Inputs:
    -------
    c: list
        A list of strings representing colors for a plot.

    index: list
        A list of indices that should be modified with the new color col.

    col: str
        The name of the new color.

    """
    
    for i in index : 
        c[i[0]] = col
     
        
def plot_behav(LC_indexes, x, U, bsr, std, color1, color2): 
    """
    Plot and save the Behavioral Saliences for specified Latent Components (LCs).

    Inputs:
    -------
    LC_indexes : list
        List  with the indexes of the Latent Variables (LVs) to plot.

    x : array-like
        Array of labels for the behavioral variables.

    U : array-like
        Matrix of behavioral saliencies.

    name : str
        Name of the saliency map.

    bsr : list
        List of integers with the stability score for U.

    std : list 
        List of integers with the standard deviation for U.

    color1 : str 
        Name of the color for bars related to non-relevant features.

    color2 : str 
        Name of the color for bars related to relevant features.

    """
    
    #shorten WarmHeartedness to WH
    if "WarmHeartedness" in x : 
        x.values[2] = "WH"

    for i in LC_indexes:  
        c = [color1]*len(U[i])
        f, ax = plt.subplots(figsize=(6,3),dpi=150)
        features_sel, selected_indexes  = boot_select(i, bsr, U)
        modify_color(c, selected_indexes, color2)
        fig=plt.bar(x, np.array(U[i]), yerr=std[i], color= c,align="center")
        ax.set_ylabel(f"Loadings {i+1}")
        plt.xticks(rotation=35, fontsize=8,ha='right')


def plot_all(res_decompo, res_permu, res_boot, type_, c_dark, c_light):
    """
    Plot behavioral data, compute a brain mask, and save NIfTI files for significant Latent Components (LCs).

    Inputs
    -------
    res_decompo : dict
        Dictionary containing decomposition results.

    res_permu : dict
        Dictionary containing permutation test results, including significant LCs.

    res_boot : dict
        Dictionary containing bootstraping results.

    type_ : str
        Type of the analysis (discrete or appraisal)

    c_dark : str
        Color representing dark tones in the plots (significant features).

    c_light : str
        Color representing light tones in the plots.
    """
    # Behavioral Data Plot
    plot_behav(res_permu['sig_LC'], res_decompo['Y'].columns, res_decompo['U'], type_,
               res_boot['bsr_u'], res_boot['u_std'], c_dark, c_light)

    # Compute brain mask from the 'gray_matter.nii.gz' file
    mask = compute_brain_mask(nib.load(os.path.join("../reg", 'gray_matter.nii.gz')))

    # Save NIfTI files for each significant LC (Latent Component)
    for LC in res_permu['sig_LC']:
        # Save NIfTI file without 0 pruning using bootstraping
        save_fMRI(np.array(res_decompo['V'][LC]), mask, f"LV{LC+1}_{type_}_all")
        
        # Prune voxels to 0 according to bootstraping results and save as NIfTI
        V_final, selected_V = boot_select(LC, res_boot['bsr_v'], res_decompo['V'])
        save_fMRI(V_final, mask, f"LV{LC+1}_{type_}")
  
        