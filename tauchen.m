function [transition_matrix, state_space] = tauchen(n, mu, rho, sigma)
% Function to implement Tauchen's method for discretizing a continuous state space
% Inputs:
% n: number of grid points
% mu: mean of the AR(1) process
% rho: AR(1) coefficient
% sigma: standard deviation of the error term
% m: number of standard deviations to approximate the state space
% Outputs:
% transition_matrix: n x n transition matrix
% state_space: n x 1 vector of state space points

m = 1/sqrt(1-rho^2);

% Compute the state space
state_space = linspace(mu - m*sigma, mu + m*sigma, n)';

% Compute the distance between grid points
d = (state_space(n) - state_space(1))/(n - 1);

% Compute the transition probabilities
transition_matrix = zeros(n, n);
for i = 1:n
    for j = 1:n
        if j == 1
            transition_matrix(i,j) = normcdf((state_space(1) - rho*state_space(i) + d/2)/sigma);
        elseif j == n
            transition_matrix(i,j) = 1 - normcdf((state_space(n) - rho*state_space(i) - d/2)/sigma);
        else
            z_low = (state_space(j) - rho*state_space(i) - d/2)/sigma;
            z_high = (state_space(j) - rho*state_space(i) + d/2)/sigma;
            transition_matrix(i,j) = normcdf(z_high) - normcdf(z_low);
        end
    end
end
end
