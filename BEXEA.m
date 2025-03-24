% BEX-SAEC: Bandit-Driven Adaptive Search in Surrogate-Assisted Evolutionary Algorithm with Explainable Uncertainty Criteria
function [hf, MaxFEs, gfs] = BEXEA(FUN, D, LB, UB)

    tic;

    % Set maximum function evaluations
    MaxFEs = 1000;
    
    dim = D;
    fobj = FUN;

    % Parameters
    T = 20;        % offspring size
    arms = 2;                         % Number of strategies
    epsilon = 0.1;                    % Epsilon for epsilon-greedy
    window_size = 20;                 % Window size

    % Training sample parameters
    if dim < 100
        N = 100;                          % < 100 dimension
    else
        N = 150;                          % >= 100 dimension
    end

    % Data archive
    Data = [];                        % Archive for surrogate model

    % Initialize tracking
    NFEs = 0;
    gfs = zeros(1, MaxFEs);         
    Q = zeros(1, arms);               % Action-value estimates
    recent_rewards = zeros(window_size, arms);
    window_idx = 1;
    best_so_far = Inf;
    stagnation_counter = 0;

    % Initialize population using LHS
    X = lhsdesign(N, dim) .* (ones(N, 1) * (UB - LB)) + ones(N, 1) * LB;
    fitness = zeros(N, 1);

    % Initial evaluations
    for i = 1:N
        fitness(i) = fobj(X(i, :));
        Data = [Data; X(i,:), fitness(i)];
        NFEs = NFEs + 1;
        best_so_far = min(best_so_far, fitness(i));
        gfs(NFEs) = best_so_far;      % Update gfs instead of cg_curve
    end

    % Initialize the reward decay factor
    decay_factor = 0.3;  % 
    reward_history = zeros(arms, 1);  % Keep track of rewards for each strategy

    % Main loop
    while NFEs < MaxFEs
        % Adaptive search behavior
        srgtOPT = srgtsRBFSetOptions(X, fitness);
        srgtSRGTRBF = srgtsRBFFit(srgtOPT);

        % Strategy selection with epsilon-greedy method
        if stagnation_counter > 20
            strategy = randi(arms);
            epsilon = min(0.3, epsilon * 1.1);
        else
            if rand < epsilon
                strategy = randi(arms);
            else
                [~, strategy] = max(Q);
            end
        end

        % Generate offspring based on strategy
        [~, best_idx] = min(fitness);
        Xbest = X(best_idx, :);
        lbest = X;
        g_best = Xbest;

        switch strategy
            case 1  % EDA
                offspring = EDA(X, LB, UB, T); 
            case 2  % QPSO
                offspring = QPSO(X, UB, LB, g_best, lbest, NFEs, MaxFEs); 
        end

        % Enhanced surrogate-based screening
        offspring_pred = rbf_predict(srgtSRGTRBF.RBF_Model, srgtSRGTRBF.P, offspring);
        eu = EUcriteria(X, fitness, offspring);
        combined_score = offspring_pred + 0.5 * (max(eu) - eu);
        [~, idx] = sort(combined_score);

        % Select and evaluate promising candidates
        num_candidates = min(3, size(offspring, 1));
        success = false;

        for k = 1:num_candidates
            if NFEs >= MaxFEs
                break;
            end

            candidate = offspring(idx(k), :);
            candidate_fitness = fobj(candidate);
            NFEs = NFEs + 1;

            % Update archive
            Data = [Data; candidate, candidate_fitness];

            % Calculate reward with decay factor and reliability consideration
            prev_best = min(fitness);
            improvement = max(0, prev_best - candidate_fitness);
            if strategy == 2
                reward = improvement / (prev_best + eps) + 0.1 * eu(idx(k));
            else
                diversity = min(pdist2(candidate, X));
                reward = improvement / (prev_best + eps) + 0.1 * diversity / dim;
            end

            % Apply decay to the reward
            reward_history(strategy) = (1 - decay_factor) * reward_history(strategy) + decay_factor * reward;
            Q(strategy) = reward_history(strategy);  % Update action-value estimate

            % Update population
            if candidate_fitness < max(fitness)
                [~, worst_idx] = max(fitness);
                X(worst_idx, :) = candidate;
                fitness(worst_idx) = candidate_fitness;
                success = true;
                stagnation_counter = 0;
            end

            % Update best-so-far and convergence curve
            prev_best_so_far = best_so_far;
            best_so_far = min(best_so_far, candidate_fitness);
            gfs(NFEs) = best_so_far;    % Update gfs instead of cg_curve

            % Display only when a new global best is found
            if candidate_fitness < prev_best_so_far
                disp(['Current optimal value = ' num2str(candidate_fitness) ' NFE=' num2str(NFEs)]);
            end

            if success && improvement > 0
                break;  % Exit if improvement found
            end
        end

        % Update stagnation counter
        if ~success
            stagnation_counter = stagnation_counter + 1;
        end
    end

    % Fill remaining convergence curve values
    if NFEs < MaxFEs
        gfs(NFEs+1:MaxFEs) = best_so_far;
    end

    % Set output for best fitness value found
    hf = best_so_far;
    
    Time = toc;
    disp(['BEXEA run time: ', num2str(Time)]);

end