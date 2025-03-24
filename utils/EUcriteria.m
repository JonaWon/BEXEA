%% Explainable uncertainty criteria
function [ y ] = EUcriteria(Xtrain, Ytrain, Xtest)
    [m, ~] = size(Xtest);
    K = 8; % number of adjacent evaluation points
    f_best = min(Ytrain); % assuming minimization problem

    y = zeros(m, 1);
    for i = 1:m
        % Find distances to all training points
        d = pdist2(Xtest(i,:), Xtrain);
        [~, ind] = sort(d);

        % Get Y values of K nearest neighbors
        nearest_Y = Ytrain(ind(1:K));

        % Predicted mean and standard deviation
        f_x = mean(nearest_Y);
        sigma_x = std(nearest_Y);

        % Calculate Expected Improvement (EI)
        if sigma_x > 0
            z = (f_x - f_best) / sigma_x;
            if f_x > f_best
                EI = (f_x - f_best) * normcdf(z) + sigma_x * normpdf(z);
            else
                EI = 0;
            end
        else
            EI = 0;
        end

        % Approximate H(SHAP(x)) using feature differences
        num_features = size(Xtrain, 2);
        feature_diff = zeros(1, num_features);
        for j = 1:num_features
            mean_feature_j = mean(Xtrain(ind(1:K), j));
            feature_diff(j) = abs(Xtest(i,j) - mean_feature_j);
        end
        total_diff = sum(feature_diff);
        if total_diff == 0
            p_j = ones(1, num_features) / num_features;
        else
            p_j = feature_diff / total_diff;
        end
        % Compute entropy
        H = 0;
        eps = 1e-10;
        for j = 1:num_features
            if p_j(j) > 0
                H = H - p_j(j) * log2(p_j(j) + eps);
            end
        end

        % Calculate the criteria S(x)
        S_x = EI / (1 + H);
        y(i) = S_x;
    end
end