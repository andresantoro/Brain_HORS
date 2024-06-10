#!/bin/sh

current_dir=`pwd`
code_dir="./../../../Core_code/"
ncores=90
output_folder="/home/asantoro/COST/Project2_Higher_Order_Brain/Brain_HORS/Figure3/Results/"
#file_input_format="TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM"



cd ${code_dir}

for subjID in `ls -d /home/asantoro/HCP/* | cut -f5 -d/` ## Loop through the subjects
do
	#echo ${subjID}
#	file_input="/home/asantoro/HCP/${subjID}/${file_input_format}_LR.mat"
	file_input="/home/asantoro/HCP/${subjID}/rfMRI_REST2_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	file_byte=`wc --bytes ${output_folder}HO_indicators_${subjID}_REST2_LR.txt | cut -d" " -f1`
	#echo ${file_byte}
	if [ ! -f ${output_folder}HO_indicators_${subjID}_REST2_LR.txt ] || [[ ${file_byte} -eq 0 ]]
	then 
		echo ${subjID} "LR"
		sh launch_julia_HOlocalinfo_all.sh ${ncores} ${file_input} ${output_folder}HO_scaffold_${subjID}_REST2_LR ${output_folder}HO_triangles_${subjID}_REST2_LR ${output_folder}HO_indicators_${subjID}_REST2_LR.txt
	fi
	
	



	file_input="/home/asantoro/HCP/${subjID}/rfMRI_REST2_RL/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat"
	file_byte=`wc --bytes ${output_folder}HO_indicatros_${subjID}_REST2_RL.txt | cut -d" " -f1`
	if [ ! -f ${output_folder}HO_indicators_${subjID}_REST2_RL.txt ] || [[ ${file_byte} -eq 0 ]]
	then
		echo ${subjID} "RL"
		sh launch_julia_HOlocalinfo_all.sh ${ncores} ${file_input} ${output_folder}HO_scaffold_${subjID}_REST2_RL ${output_folder}HO_triangles_${subjID}_REST2_RL ${output_folder}HO_indicators_${subjID}_REST2_RL.txt
	fi
done

cd ${current_dir}

