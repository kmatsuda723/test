function [transition_matrix, state_space] = rouwenhorst(n, mu, rho, sigma)
% Function to implement the Rouwenhorst method for discretizing a continuous state space
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

% Compute the transition probabilities using the Rouwenhorst method
P = zeros(n, n);
for i = 1:n
    P(i,1) = normcdf((state_space(1) - rho*state_space(i) + (state_space(2)-state_space(1))/2)/sigma);
    P(i,n) = 1 - normcdf((state_space(n) - rho*state_space(i) - (state_space(n)-state_space(n-1))/2)/sigma);
    for j = 2:n-1
        z_low = (state_space(j) - rho*state_space(i) + (state_space(j)-state_space(j-1))/2)/sigma;
        z_high = (state_space(j) - rho*state_space(i) - (state_space(j+1)-state_space(j))/2)/sigma;
        P(i,j) = normcdf(z_high) - normcdf(z_low);
    end
end
transition_matrix = (eye(n) - rho*P) + (1/n)*(1-rho)*ones(n);

end