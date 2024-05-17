#!/bin/bash

current_dir=`pwd`
code_dir="./../../../Core_code/"
ncores=94
output_folder="/home/asantoro/COST/Project2_Higher_Order_Brain/Brain_HORS/Figure2/Results/HO_indicators/"
file_input_format="TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM"
tasklist=( "GAMBLING" "LANGUAGE" "MOTOR" "RELATIONAL" "SOCIAL" "WM" ) 

#file_input="/home/asantoro/HCP/100307/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"

cd ${code_dir}



for subjID in `ls -d /home/asantoro/HCP/* | cut -f5 -d/` ## Loop through the subjects
do
	echo ${subjID}
	file_input="/home/asantoro/HCP/${subjID}/${file_input_format}_LR.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HOindicators_${subjID}_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_LR
		
	file_input="/home/asantoro/HCP/${subjID}/${file_input_format}_RL.mat"
	sh launch_julia_HOindicators_only.sh ${ncores} ${file_input} ${output_folder}HOindicators_${subjID}_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_RL
	done
cd ${current_dir}

