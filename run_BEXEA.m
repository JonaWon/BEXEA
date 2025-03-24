function [gsamp1 ,time_cost] = run_BEXEA(runs, D, FUN, LB, UB, fname)

    time_begin=tic;
    warning('off');
    addpath(genpath(pwd));
    
    for r=1:runs
        % main loop
        fprintf('\n');
        disp(['FUNCTION: ', fname,' RUN: ', num2str(r)]);  
        fprintf('\n');
        [hisf,mf,gfs]= BEXEA(FUN,D,LB,UB); 
        fprintf('Best fitness (final): %e\n',min(hisf));       
        gsamp1(r,:)=gfs(1:mf);
    end    
    
    % Running time
    time_cost=toc(time_begin);

end
    