#!/bin/sh

current_dir=`pwd`
code_dir="./../../../Core_code/"
ncores=80
output_folder="/home/asantoro/COST/Project2_Higher_Order_Brain/Brain_HORS/Figure2/Results/HO_indicators/"


#file_input="/home/asantoro/HCP/100307/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"

cd ${code_dir}

for subjID in `ls -d /home/asantoro/HCP/* | cut -f5 -d/` ## Loop through the subjects
do
	echo ${subjID}
	file_input="/home/asantoro/HCP/${subjID}/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HO_indicators_${subjID}_REST1_LR.txt
	
	file_input="/home/asantoro/HCP/${subjID}/rfMRI_REST1_RL/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HO_indicators_${subjID}_REST1_RL.txt
done

cd ${current_dir}
