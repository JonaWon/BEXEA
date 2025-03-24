%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for the BEXEA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
addpath(genpath(pwd));

% Experiment Parameter
test_func = {'CEC2022-F1','CEC2022-F2','CEC2022-F3','CEC2022-F4','CEC2022-F5','CEC2022-F6','CEC2022-F7','CEC2022-F8','CEC2022-F9','CEC2022-F10','CEC2022-F11','CEC2022-F12'} %#ok<NOPTS>

dim = 20;    % Dimensions 
lb = -100;
ub = 100;
Runs = 20;    % Number of runs

for j = 1:size(test_func, 2)
    func_name = test_func{j};                  
    FUN = @(x) cec22_func(x', j);
    
    LB = repmat(lb, 1, dim);             
    UB = repmat(ub, 1, dim);
    
    fprintf('Running D=%d, Function=%s\n', dim, func_name);
    [gsamp1,time_cost] = run_BEXEA(Runs,dim,FUN, LB, UB, func_name);
end

save Result                     