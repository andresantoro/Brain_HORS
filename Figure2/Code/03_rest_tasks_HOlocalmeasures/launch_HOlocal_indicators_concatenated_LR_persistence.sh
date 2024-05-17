#!/bin/sh

current_dir=`pwd`
code_dir="./../../../Core_code/"
ncores=90
output_folder="/home/asantoro/COST/Project2_Higher_Order_Brain/Brain_HORS/Figure2/Results/HO_localmeasures/"
file_input_format="TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM"


cd ${code_dir}

for subjID in `ls -d /home/asantoro/HCP/* | cut -f5 -d/` ## Loop through the subjects
do
	echo ${subjID}
	file_input="/home/asantoro/HCP/${subjID}/${file_input_format}_LR.mat"
	sh launch_julia_HOlocalinfo_onlyscaffold_persistence.sh ${ncores} ${file_input} ${output_folder}HO_scaffold_persistence_${subjID}_LR
done

cd ${current_dir}

