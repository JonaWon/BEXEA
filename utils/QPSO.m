function [POP] = QPSO(POP, VRmax, VRmin, g_best, lbest, i, me)
    % INPUT:
    %   POP    : Population matrix (size: ps × (D + auxiliary data))
    %   VRmax  : Upper bounds of variables (1 × D)
    %   VRmin  : Lower bounds of variables (1 × D)
    %   g_best : Global best position (1 × D)
    %   lbest  : Local best positions (ps × D)
    %   i      : Current iteration
    %   me     : Maximum iterations
    % OUTPUT:
    %   POP    : Updated population matrix with new positions

    % Population size (number of particles)
    ps = size(POP,1);  
    % Dimensionality of the problem
    D = size(VRmax,2);
    % Extract particle positions
    pos = POP(:,1:D);

    % Local and global best positions
    pbest = lbest(:,1:D);
    gbest=g_best(:,1:D);
    gbestrep = repmat(gbest, ps, 1);  % Repeat global best for all particles

    % Mean best position (average of local bests)
    pavg = mean(pbest);
    pavgrep = repmat(pavg, ps, 1);  % Repeat mean best for all particles

    % Random coefficients for quantum behavior
    fi = rand(ps, D);
    p = fi .* pbest + (1 - fi) .* gbestrep;

    % Adaptive contraction-expansion coefficient
    alpha = (1 - 0.5) * (me - i) / me + 0.5;  

    % Compute step size (quantum potential-based search)
    b = alpha * abs(pavgrep - pos);
    u = rand(ps, D);
    
    % New position update using QPSO equation
    pos = p + ((-1).^ceil(0.5 + rand(ps, D))) .* b .* (-log(u));

    % Handle boundary constraints
    pos = ((pos >= VRmin) & (pos <= VRmax)) .* pos + ...
          ((pos < VRmin) | (pos > VRmax)) .* (VRmin + (VRmax - VRmin) .* rand(ps, D));

    % Update positions in population matrix
    POP(:,1:D) = pos;
end