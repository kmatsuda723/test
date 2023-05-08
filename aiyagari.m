clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set parameter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigma = 1.50; % risk aversion
beta = 0.98; % subjective discount factor
delta = 0.03; % depreciation
Z = 1.00; % production technology
alpha = 0.25; % capital’s share of income
Kstart = 10.0; % initial value for aggregate capital stock
g = 0.2; % iteration relaxation parameter
rho = 0.6; % labor productivity persistence
sigma_eps = sqrt(0.6*(1-rho^2)); % labor productivity variance

% form labor productivity grids

NS = 2;
[prob, eta] = tauchen(NS, -0.7, rho, sigma_eps); % Discretization of AR(1) prod process
eta = exp(eta);

% form capital grids

a_l = 0; % minimum value of capital a
a_u = 20; % maximum value of capital a
inckap = 0.05; % size of capital a increments
NA = round((a_u-a_l)/inckap+1); % number of a points
a = linspace(a_l, a_u, NA)'; % grids

% calculate aggregate labor supply in steady state

[v1,d1]=eig(prob'); % eigenvalues
[dmax,imax]=max(diag(d1));
probst1=v1(:,imax);
ss=sum(probst1);
probst1=probst1/ss; % ss distribution
HH = sum(eta.*probst1); % aggregate effective labor

% simulations of shocks

Nsim = 1;
Tsim = 50;
for i = 1:Nsim
    is_t(i, :) = simulate_markov(Tsim, probst1, prob);
end

% redistribution policy

trans = 0.1;
taul = trans/HH;

% loop to find fixed point for agregate capital stock

liter = 1;
itermax = 50;
toler = 0.001; % warning: this doens't converge if tolerance is too small
metric = 10; % initial difference
KK = Kstart; % initial capital
disp('ITERATING ON KK');
disp('');
disp('     liter    metric     meanK      Kold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop for finding eq capital
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (metric > toler) && (liter <= itermax)

    % calculate rental rate of capital and w

    w = (1-alpha) * Z * KK^(alpha) * HH^(-alpha);% Derive w and r from eq capital and labor
    r = (alpha) * Z * KK^(alpha-1) * HH^(1-alpha);



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solving for households optimization (policy function of assets)
    % choose one of them
    % 1. grid search
    % 2. bisection minimization
    % 3. first order condition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % aplus = solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans); 
    % aplus = solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans);
    aplus = solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop for finding eq distribution and capital
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % eq distribution

    phi = get_distribution(aplus, a_l, a_u, NA, NS, prob);

    % new aggregate capital

    meanK = sum(phi.*aplus,'all');

    % capital holdings distribution

    probk = sum(phi, NS);

    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     % Loop for finding eq capital with simulations
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %     a_t = zeros(Nsim, Tsim+1);
    %     aplus_t = zeros(Nsim, Tsim+1);
    %     c_t = zeros(Nsim, Tsim);
    %
    %     a_t(:, 1) = KK;
    %     for i = 1:Nsim
    %         for it = 1:Tsim
    %             [ial(i, it), iar(i, it), varphi(i, it)] = linint(a_t(i, it), a_l, a_u, NA);
    %             a_t(i, it+1) = varphi(i, it)*aplus(ial(i, it), is_t(i, it)) + (1-varphi(i, it))*aplus(iar(i, it), is_t(i, it));
    %             aplus_t(i, it) = a_t(i, it+1);
    %             c_t(i, it) = w*eta(is_t(i, it)) + (1 + r - delta)*a_t(i, it) - aplus_t(i, it);
    %         end
    %     end
    %
    %     meanK = sum(a_t(:, Tsim))/Nsim;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop for finding eq capital
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %
    % form metric and update KK
    %
    Kold = KK;
    Knew = g*meanK + (1-g)*Kold;
    metric = abs((Kold-meanK)/Kold);

    KK = Knew;
    disp([ liter metric meanK Kold]);
    liter = liter+1;

end
%
% print out results
%
disp('PARAMETER VALUES');
disp('');
disp('     sigma      beta     delta         Z     alpha');
disp([ sigma beta delta Z alpha]);
disp('');
disp('EQUILIBRIUM RESULTS ');
disp('');
disp('        KK        HH      w      r');
disp([ Kold HH w r ]);
%
% simulate life histories of the agent
%
disp('SIMULATING LIFE HISTORY');

% initial

a_t = zeros(Nsim, Tsim+1);
aplus_t = zeros(Nsim, Tsim+1);
c_t = zeros(Nsim, Tsim);
a_t(:, 1) = KK;

% simulation

for i = 1:Nsim % agents
    for it = 1:Tsim % time
        [ial(i, it), iar(i, it), varphi(i, it)] = linint(a_t(i, it), a_l, a_u, NA);
        a_t(i, it+1) = varphi(i, it)*aplus(ial(i, it), is_t(i, it)) + (1-varphi(i, it))*aplus(iar(i, it), is_t(i, it));
        aplus_t(i, it) = a_t(i, it+1);
        c_t(i, it) = (1-taul)*w*eta(is_t(i, it)) + trans + (1 + r - delta)*a_t(i, it) - aplus_t(i, it);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2,2,1),plot((1:Tsim)',a_t(1, 2:Tsim+1)-a_t(1, 1:Tsim),(1:Tsim)',c_t(1, :));
title('MODEL 2: INVESTMENT AND CONSUMPTION');
disp('Covariance matrix');
disp([cov(c_t(1, :),a_t(1, 2:Tsim+1)-a_t(1, 1:Tsim))]);
%
% calculate income distribution %
income = [ (r*a + w*eta(1)) (r*a + w*eta(2)) ] ;
[ pinc, index ] = sort(income(:));
plambda = phi(:);
%
subplot(2,2,2), plot(pinc,plambda(index));
title('MODEL 2: INCOME DISTRIBUTION');
xlabel('INCOME LEVEL');
ylabel('% OF AGENTS');
%
% calculate capital distribution
%
subplot(2,2,3),plot(a,probk);
title('MODEL 2: CAPITAL DISTRIBUTION');
xlabel('CAPITAL a');
ylabel('% OF AGENTS');
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid search
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function aplus = solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)

% tabulate the utility function such that for zero or negative
% consumption utility remains a large negative number so that
% such values will never be chosen as utility maximizing

a = linspace(a_l, a_u, NA)'; % create a grid of asset holdings
util = -10000*ones(NA, NA, NS); % initialize the utility function to a large negative number for zero or negative consumption

for i=1:NA
    kap=a(i);
    for j=1:NA
        kapp = a(j);
        for is = 1:NS
            cons = (1-taul)*w*eta(is) + trans + (1 + r - delta)*kap - kapp; % calculate consumption for each asset combination and shock
            if cons > 0 % if consumption is positive
                util(j, i, is)=(cons)^(1-sigma)/(1-sigma); % calculate utility and store it in the utility function matrix
            end
        end
    end
end

% initialize some variables %
v = zeros(NA, NS);
aplus = zeros(NA, NS);
test = 10;
[rs,cs] = size(util(:, :, 1)); %
r = zeros(NA, NA, NS);
% iterate on Bellman’s equation and get the decision
% rules and the value function at the optimum %
while test ~= 0 % while the difference between the new and old policy function is not zero
    for i=1:cs
        for is = 1:NS
            r(:,i, is)=util(:,i, is)+beta*(v(:, :)*prob(is, :)'); % calculate the expected value of next period's utility function



        end
    end


    for is = 1:NS
        [tv_help(is, :), tdecis_help(is, :)]=max(r(:, :, is));
    end

    tdecis=tdecis_help(:, :)'; % reshape the optimal asset choice array
    tv=tv_help(:, :)'; % reshape the expected value array

    test=max(any(tdecis-aplus)); % calculate the difference between the new and old policy function
    v=tv; % update the expected value array
    aplus=tdecis; % update the policy function

end
aplus=a(aplus); % map the policy function indices to actual asset values
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interpolated value function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function aplus = solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)

% Grid of asset holdings
a = linspace(a_l, a_u, NA)';

% Initialize variables
EV = -82*ones(NA, NS);
EV_new = zeros(NA, NS);
v = zeros(NA, NS);
aplus = zeros(NA, NS);
test = 10;

% iterate on Bellman’s equation and get the decision
% rules and the value function at the optimum %
while test > 10^(-4)
    for is = 1:NS
        % use fminbnd to solve for optimal asset holding
        parfor ia = 1:NA
            % the anonymous function takes in the current asset level a(ia)
            % and returns the negative of the value function v(a_plus) to be minimized
            [aplus(ia, is), v(ia, is)] = fminbnd(@(x) -valuefunc(x, is, EV, (1-taul)*w*eta(is) + trans + (1 + r - delta)*a(ia), beta, sigma, a_l, a_u, NA), a_l, a_u);
            v(ia, is) = -v(ia, is);
        end
    end
    for is = 1:NS
        % calculate the new value function by taking the expectation of the
        % value function at the optimal asset level for each state tomorrow
        EV_new(:, is) = v(:, :)*prob(is, :)';
    end
    % calculate the test statistic for convergence
    test = max(abs(EV_new - EV), [], 'all');
    EV = EV_new; % update value function
end



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization with first order condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function aplus = solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)

% Grid of asset holdings
a = linspace(a_l, a_u, NA)';

% Initialize variables
RHS = zeros(NA, NS);
c = zeros(NA, NS);

% Compute current consumption
for is = 1:NS
    c(:, is) = a(:) + eta(is);
end
c_new(:, :) = c(:, :);

% Set initial test value for convergence
test = 10;

% Loop until convergence is reached
while test > 10^(-4)

    % Compute the RHS of the Euler equation
    RHS(:, :) = (c_new(:, :).^(-sigma))*prob(:, :)';
    RHS(:, :) = (beta*(1+r-delta)*RHS(:, :)).^(-1/sigma);

    % Solve for optimal consumption using fzero
    for is = 1:NS
        parfor ia = 1:NA
            % Solve the first-order condition for consumption
            c_new(ia, is) = fzero(@(x) foc(x, is, (1-taul)*w*eta(is) + trans + (1 + r - delta)*a(ia), RHS, sigma, a_l, a_u, NA), c(ia, is));
        end
    end

    % Implement constraints
    c_new = min(c_new, (1-taul)*w*repmat(eta', NA, 1) + trans + (1 + r - delta)*repmat(a, 1, NS) - a_l);
    c_new = max(c_new, 1e-4);

    % Calculate convergence criterion
    test = max(abs(c_new - c), [], 'all')./max(abs(c), [], 'all');

    % Update c using a dampened update rule
    c = 0.2*c_new + 0.8*c;

    % Calculate optimal future assets
    aplus = w*repmat(eta', NA, 1) + (1 + r - delta)*repmat(a, 1, NS) - c;

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First order condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function foc_v = foc(x_in, is, available, RHS, sigma, a_l, a_u, NA)

% future assets
a_plus = available - x_in;

% linear interpolation
[ial, iar, varphi] = linint(a_plus, a_l, a_u, NA);
foc_v = varphi*RHS(ial, is) + (1-varphi)*RHS(iar, is);

% get first order condition
foc_v = x_in - foc_v;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Value function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function obj = valuefunc(a_plus, is, EV, available, beta, sigma, a_l, a_u, NA)
% Computes the value function for a given interest rate state (is) and
% choice of additional assets (a_plus) given the current state variables.
% The value function is defined as the sum of the immediate utility (cons)
% and the discounted expected value of future utility (EV), minus a large
% penalty term for infeasible choices of a_plus.
%
% Inputs:
% a_plus: vector of additional assets to choose from
% is: current interest rate state
% EV: expected value function for the next period
% available: current amount of available assets
% beta: discount factor
% sigma: coefficient of relative risk aversion
% a_l, a_u: lower and upper bounds for the grid of asset values
% NA: number of points on the grid of asset values
%
% Output:
% obj: value function for the given interest rate state and choice of
% additional assets.

a = linspace(a_l, a_u, NA)'; % Generate evenly spaced grid of asset values

cons = max(available - a_plus, 1e-10); % Compute consumption

% Compute the interpolated indices and weights for linear interpolation
% of the expected value function
[ial, iar, varphi] = linint(a_plus, a_l, a_u, NA);

% Compute the objective function as the sum of immediate utility, the
% discounted expected value of future utility, and a large penalty term
% for infeasible choices of a_plus.
obj = varphi*EV(ial, is) + (1-varphi)*EV(iar, is);
obj = (cons)^(1-sigma)/(1-sigma) + beta*obj - 100000*abs(cons - available + a_plus);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distribution iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function phi = get_distribution(aplus, a_l, a_u, NA, NS, prob)

a = linspace(a_l, a_u, NA)'; % Generate a linearly spaced vector of NA values between a_l and a_u

ial = zeros(NA, NS); % Initialize the indices of aplus values rounded down
iar = zeros(NA, NS); % Initialize the indices of aplus values rounded up
varphi = zeros(NA, NS); % Initialize the blending coefficients

for ia = 1:NA % Loop over all values of a
    for is = 1:NS % Loop over all values of s
        [ial(ia, is), iar(ia, is), varphi(ia, is)] = linint(aplus(ia, is), a_l, a_u, NA); % Call the linint function to interpolate aplus values to get ial, iar, and varphi values
        varphi(ia, is) = max(min(varphi(ia, is), 1), 0); % Clip varphi values to be within [0, 1]
    end
end

test = 10; % Initialize a test value to be greater than 10^-8
phi = ones(NA, NS)/NA/NS; % Initialize the distribution phi to be uniform

while test > 10^(-8) % Loop until the test value is less than 10^-8
    phi_new = zeros(NA, NS); % Initialize a new distribution phi_new to be all zeros
    for ia = 1:NA % Loop over all values of a
        for is = 1:NS % Loop over all values of s
            for is_p = 1:NS % Loop over all values of s'
                phi_new(ial(ia, is), is_p) = phi_new(ial(ia, is), is_p) + prob(is, is_p)*varphi(ia, is)*phi(ia, is); % Update phi_new using the interpolation indices, blending coefficients, and probabilities
                phi_new(iar(ia, is), is_p) = phi_new(iar(ia, is), is_p) + prob(is, is_p)*(1-varphi(ia, is))*phi(ia, is);
            end
        end
    end
    test=max(abs(phi_new-phi), [], 'all'); % Calculate the maximum difference between phi_new and phi
    phi = phi_new; % Update phi to be phi_new
end


end



