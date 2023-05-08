function [states, pi] = simulate_markov(T, pi0, pi)

    % T: length of time series
    % pi0: initial distribution over states
    % pi: transition matrix
    
    % Initialize the random seed
    rng('shuffle');
%     rng(1);
    
    % Determine the number of states
    N = size(pi, 1);
    
    % Initialize the state sequence
    states = zeros(T, 1);
    
    % Draw the initial state from pi0
    states(1) = find(mnrnd(1, pi0));
    
    % Generate the state sequence
    for t = 2:T
        states(t) = find(mnrnd(1, pi(states(t-1), :)));
    end
    
    % Compute the empirical transition matrix
    pi_emp = zeros(N, N);
    for i = 1:N
        pi_emp(i, :) = sum(states(1:end-1) == i & states(2:end) == (1:N), 1);
        pi_emp(i, :) = pi_emp(i, :) / sum(pi_emp(i, :));
    end
    
    % Return the empirical transition matrix
    pi = pi_emp;

end
