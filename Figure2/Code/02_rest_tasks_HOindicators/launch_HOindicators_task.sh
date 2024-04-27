#!/bin/sh

current_dir=`pwd`
code_dir="./../../../Core_code/"
ncores=82
output_folder="/home/asantoro/COST/Project2_Higher_Order_Brain/Brain_HORS/Figure2/Results/HO_indicators/"


#file_input="/home/asantoro/HCP/100307/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"

cd ${code_dir}

taskname="EMOTION"

for subjID in `ls -d /home/asantoro/HCP/* | cut -f5 -d/` ## Loop through the subjects
do
	echo ${subjID}
	file_input="/home/asantoro/HCP/${subjID}/tfMRI_${taskname}_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HO_indicators_${subjID}_${taskname}_LR.txt
	
	file_input="/home/asantoro/HCP/${subjID}/tfMRI_${taskname}_RL/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HO_indicators_${subjID}_${taskname}_RL.txt
done

cd ${current_dir}
