%% This function constructs the 'paradigm' of interest
function [time,Regressor] = GSP_Paradigm(PathToPar,n_rem,task)
    
    
    Now = pwd;
    cd(PathToPar);
%     n_shift = 10;
    % Creates the regressor that will highlight the presentation of the
    % stimulus
    
    
    switch task
        case 'tfMRI_GAMBLING'
            Regressor = zeros(1,253);
            time = 0:0.72:(253*0.72);
            time = time(1:end-1);
            
            %loss
            tmp = textread('loss.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 1;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 1;
            % win
            tmp = textread('win.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 2;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 2;
            
        case 'tfMRI_WM'
            Regressor = zeros(1,405);
            time = 0:0.72:(405*0.72);
            time = time(1:end-1);
            
            % 0-back body
            tmp = textread('0bk_body.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 1;
            % 0-back faces
            tmp = textread('0bk_faces.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 2;
            % 0-back places
            tmp = textread('0bk_places.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 3;
            % 0-back tools
            tmp = textread('0bk_tools.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 4;
            % 2-back body
            tmp = textread('2bk_body.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 5;
            % 2-back faces
            tmp = textread('2bk_faces.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 6;
            % 2-back tools
            tmp = textread('2bk_places.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 7;
            % 2-back places
            tmp = textread('2bk_tools.txt');
            Regressor(ceil(tmp(1)/0.72):(ceil(tmp(1)/0.72)+floor(tmp(2)/0.72))) = 8;
        case 'tfMRI_MOTOR'
            Regressor = zeros(1,284);
            time = 0:0.72:(284*0.72);
            time = time(1:end-1);
            
            
            % left foot
            tmp = textread('lf.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 1;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 1;
            
            % right foot
            tmp = textread('rf.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 2;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 2;
            
            % right hand
            tmp = textread('rh.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 3;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 3;
             
            % left hand
            tmp = textread('lh.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 4;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 4;
           
%              % tongue
            tmp = textread('t.txt');
            Regressor(ceil(tmp(1,1)/0.72):(ceil(tmp(1,1)/0.72)+floor(tmp(1,2)/0.72))) = 5;
            Regressor(ceil(tmp(2,1)/0.72):(ceil(tmp(2,1)/0.72)+floor(tmp(2,2)/0.72))) = 5;
    
        case 'tfMRI_LANGUAGE'
            Regressor = zeros(1,316);
            time = 0:0.72:(316*0.72);
            time = time(1:end-1);
            
            % math
            tmp = textread('math.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 1;
            end
            
            % story
            tmp = textread('story.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 2;
            end
            
        case 'tfMRI_RELATIONAL'
            Regressor = zeros(1,232);
            time = 0:0.72:(232*0.72);
            time = time(1:end-1);
            
            % relation
            tmp = textread('relation.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 1;
            end
            
            % match
            tmp = textread('match.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 2;
            end
        case 'tfMRI_EMOTION'
            
            Regressor = zeros(1,176);
            time = 0:0.72:(176*0.72);
            time = time(1:end-1);
            
            
            % fear
            tmp = textread('fear.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 1;
            end
            
            % neutral
            tmp = textread('neut.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 2;
            end
            
        case 'tfMRI_SOCIAL'
            Regressor = zeros(1,274);
            time = 0:0.72:(274*0.72);
            time = time(1:end-1);
            
            
            
            tmp = textread('mental.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 1;
            end
            
              
            
            tmp = textread('rnd.txt');
            for k = 1:size(tmp,1)
                Regressor(ceil(tmp(k,1)/0.72):(ceil(tmp(k,1)/0.72)+floor(tmp(k,2)/0.72))) = 2;
            end
            
    end
    Regressor(1:n_rem) = [];
    time(1:n_rem) = [];
    
    cd(Now);
end