%% List all the 100 subjects and loop through them
current_dir=pwd;
list_subj_ID=dir('/home/andrea/miplabsrv2/HCP/*');
list_subj_ID(1)=[];
list_subj_ID(1)=[];
%min_length_task={1000,1000,1000,1000,1000,1000,1000};
min_length_task={176,253,316,284,232,274,405};
load('min_size_tasks_regressors.mat')
%min_size_tasks_regressors=cell(200,7);
for k_ID= 1:100
    subj_ID=list_subj_ID(k_ID).name;
    %disp(list_subj_ID(k_ID));
    data_rest1_LR=load(['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/rfMRI_REST1_LR/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat']).TS;
    data_rest1_RL=load(['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/rfMRI_REST1_RL/Schaefer100/TS_Schaefer100S_gsr_bp_z.mat']).TS;

    %% Take only 300 time points for rest (so that they are comparable with length of the tasks)
    data_rest1_LR=data_rest1_LR(:,1:300);
    data_rest1_RL=data_rest1_RL(:,1:300);

    %% List of tasks
    task_list={'EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM'};
    shift_skip={0,0,0,0,0,0,0};
    data_task=cell([1,length(task_list)]);
    LR_RL_flag={'LR','RL'};
    for s=1:2
        for i=1:length(task_list)
            
            %Path with the task regressors
            path_EV=['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/tfMRI_',task_list{i},'_',LR_RL_flag{s},'/EVs/'];
            %Path of the task fMRI data 
            path_task=['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/tfMRI_',task_list{i},'_',LR_RL_flag{s},'/Schaefer100/TS_Schaefer100S_gsr_bp_z'];
            %disp([task_list{i},' ',path_EV,' ',path_task]);   
            
            c_TS_task=load(path_task).TS;
            %data_task{1,i}=c_TS_task;
            taskID=['tfMRI_',task_list{i}];
            [time,Regressor]=GSP_Paradigm(path_EV,shift_skip{i},taskID);
            %% If the regressor has more points than the recording, then I trim the length of the regressor
            Regressor=Regressor(1:min_length_task{i});
            c_TS_task=c_TS_task(:,1:min_length_task{i});
            %Regressor=Regressor(1:length(c_TS_task));
%             if min_length_task{i}> length(c_TS_task)
%                 min_length_task{i}=length(c_TS_task);
%             end
            
            %% Find indexes of all the task regressors with the same order
            all_indexes_regressors=[];
            m = max(Regressor);
            vec_size_indexes=[];
            for c_id=1:m
                c_indexes=find(Regressor==c_id);
                %%This is fixing the length of different regressors (mainly
                %%for the language task)
                %%Taking the minimum index across subjects
                current_min_indexes_across_subjects=min(cell2mat(min_size_tasks_regressors(:,i)));
                final_index=current_min_indexes_across_subjects(c_id);
                all_indexes_regressors=[all_indexes_regressors,c_indexes(1:final_index)];
                
                %vec_size_indexes=[vec_size_indexes,length(c_indexes)];
                %all_indexes_regressors=[all_indexes_regressors,c_indexes];
            end
            %min_size_tasks_regressors{2*(k_ID-1)+s,i}=vec_size_indexes;
            
            TS_onlytask=c_TS_task(:,all_indexes_regressors);
            data_task{1,i}=TS_onlytask;
            disp([subj_ID,' ',task_list{i},' ',num2str(length(Regressor)),' ',num2str(length(c_TS_task)),'--->',num2str(length(all_indexes_regressors))]);
        end
        %%Concatenate rest (300) + 7 HCP task in order (LR and RL separately)
        if s==1
            TS=data_rest1_LR;
        else
            TS=data_rest1_RL;
        end
        for l=1:length(task_list)
            TS=[TS,data_task{1,l}];
        end
        disp([subj_ID,' ', LR_RL_flag{s},' ']);
        disp(size(TS));

        if s==1
            output_path=['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_LR.mat'];
            %save(output_path,'TS');
        else
            output_path=['/home/andrea/miplabsrv2/HCP/',num2str(subj_ID),'/TS_Schaefer100S_gsr_bp_z_rest300_tasks_EMO_GAM_LAN_MOT_REL_SOC_WM_RL.mat'];
            %save(output_path,'TS');
        end
    end
end