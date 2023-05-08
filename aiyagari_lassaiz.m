clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set parameter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigma = 1.50; % risk aversion
beta = 0.98; % subjective discount factor
delta = 0.03; % 1 - depreciation
Z = 1.00; % production technology
alpha = 0.25; % capital’s share of income
% theta = 0.05; % non-rental income if unemployed is theta*w
Kstart = 10.0; % initial value for aggregate capital stock
g = 0.2; % iteration relaxation parameter

rho = 0.6;
sigma_eps = sqrt(0.6*(1-rho^2));

NS = 2;

[prob, eta] = tauchen(NS, -0.7, rho, sigma_eps);
eta = exp(eta);
%
% eta = zeros(NS, 1);
% eta(1) = 0.05;
% eta(2) = 1;
% prob = [ .5 .5;  .2 .8]; % prob(i,j) = probability (s(t+1)=sj | s(t) = si)


% form capital a

a_l = 0;
a_u = 20; % maximum value of capital a
inckap = 0.05; % size of capital a increments
NA = round((a_u-a_l)/inckap+1); % number of a points
a = linspace(a_l, a_u, NA)';

% calculate aggregate labor supply

[v1,d1]=eig(prob');
[dmax,imax]=max(diag(d1));
probst1=v1(:,imax);
ss=sum(probst1);
probst1=probst1/ss;
HH = sum(eta.*probst1);%eta(1)*pempl + eta(2)*(1-pempl);


Nsim = 100;
Tsim = 100;
for i = 1:Nsim
    is_t(i, :) = simulate_markov(Tsim, probst1, prob);
end

% loop to find fixed point for agregate capital stock

liter = 1;
itermax = 50;
toler = 0.001;% warning: this doens't converge if tolerance is too small
metric = 10;
KK = Kstart;
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

%     aplus = solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS);
    %         aplus = solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS);
        aplus = solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop for finding eq capital
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        phi = get_distribution(aplus, a_l, a_u, NA, NS, prob);
    
        meanK = sum(phi.*aplus,'all');
    
    
        probk = sum(phi, NS);




%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Loop for finding eq capital
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     a_t = zeros(Nsim, Tsim+1);
%     aplus_t = zeros(Nsim, Tsim+1);
%     c_t = zeros(Nsim, Tsim);
% 
%     a_t(:, 1) = KK;
% % 
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
    disp([ liter metric meanK Kold ]);
    liter = liter+1;

end
%
% print out results
%
disp('PARAMETER VALUES');
disp('');
disp('     sigma      beta     delta         Z     alpha     eta');
disp([ sigma beta delta Z alpha eta']);
disp('');
disp('EQUILIBRIUM RESULTS ');
disp('');
disp('        KK        HH      w      r');
disp([ Kold HH w r ]);
%
% simulate life histories of the agent
%
disp('SIMULATING LIFE HISTORY');

    a_t = zeros(Nsim, Tsim+1);
    aplus_t = zeros(Nsim, Tsim+1);
    c_t = zeros(Nsim, Tsim);

    a_t(:, 1) = KK;

    for i = 1:Nsim
        for it = 1:Tsim
            [ial(i, it), iar(i, it), varphi(i, it)] = linint(a_t(i, it), a_l, a_u, NA);
            a_t(i, it+1) = varphi(i, it)*aplus(ial(i, it), is_t(i, it)) + (1-varphi(i, it))*aplus(iar(i, it), is_t(i, it));
            aplus_t(i, it) = a_t(i, it+1);
            c_t(i, it) = w*eta(is_t(i, it)) + (1 + r - delta)*a_t(i, it) - aplus_t(i, it);
        end
    end


subplot(2,2,1),plot((1:Tsim)',a_t(1, 2:Tsim+1)-a_t(1, 1:Tsim),(1:Tsim)',c_t(1, :));
title('MODEL 2: INVESTMENT AND CONSUMPTION');
% print histmod2
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
% print distmod2
%
% calculate capital distribution
%
subplot(2,2,3),plot(a,probk);
title('MODEL 2: CAPITAL DISTRIBUTION');
xlabel('CAPITAL a');
ylabel('% OF AGENTS');
% print capdmod2
%%%%%%%%%%%%%%%%%%%%%%%%

function aplus = solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS)

% tabulate the utility function such that for zero or negative
% consumption utility remains Z large negative number so that
% such values will never be chosen as utility maximizing
%

a = linspace(a_l, a_u, NA)';

for is = 1:NS
    util(:, :, is) = -10000*ones(NA,NA);
end
for i=1:NA
    kap=a(i);% 効用関数の作り方は離散近似
    for j=1:NA
        kapp = a(j);
        for is = 1:NS
            cons = w*eta(is) + (1 + r - delta)*kap - kapp;
            if cons > 0
                util(j, i, is)=(cons)^(1-sigma)/(1-sigma);
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
while test ~= 0
    for i=1:cs
        for is = 1:NS
            r(:,i, is)=util(:,i, is)+beta*(v(:, :)*prob(is, :)');
        end
    end
    for is = 1:NS
        [tv_help(is, :), tdecis_help(is, :)]=max(r(:, :, is));
    end

    tdecis=tdecis_help(:, :)';
    tv=tv_help(:, :)';

    test=max(any(tdecis-aplus));
    v=tv;
    aplus=tdecis;
end
aplus=a(aplus);
end


function aplus = solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS)

% tabulate the utility function such that for zero or negative
% consumption utility remains Z large negative number so that
% such values will never be chosen as utility maximizing

a = linspace(a_l, a_u, NA)';

% initialize some variables %
EV = -82*ones(NA, NS);
EV_new = zeros(NA, NS);
v = zeros(NA, NS);

aplus = zeros(NA, NS);
test = 10;

% iterate on Bellman’s equation and get the decision
% rules and the value function at the optimum %
while test > 10^(-4)
    for is = 1:NS
        parfor ia = 1:NA
            %             [aplus(ia, is), v(ia, is)] = golden_search(@(x) -valuefunc(x, is, EV, w*eta(is) + (r + delta)*a(ia), beta, sigma, a_l, a_u, NA), a_l, a_u);
            [aplus(ia, is), v(ia, is)] = fminbnd(@(x) -valuefunc(x, is, EV, w*eta(is) + (1 + r - delta)*a(ia), beta, sigma, a_l, a_u, NA), a_l, a_u)
            v(ia, is) = -v(ia, is);
        end
    end
    for is = 1:NS
        EV_new(:, is) = v(:, :)*prob(is, :)';
    end
    test = max(abs(EV_new - EV), [], 'all');
    EV = EV_new;
end

end

function aplus = solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS)

a = linspace(a_l, a_u, NA)';
RHS = zeros(NA, NS);
c = zeros(NA, NS);

for is = 1:NS
    c(:, is) = a(:) + eta(is);
end
c_new(:, :) = c(:, :);

test = 10;

while test > 10^(-4)

    RHS(:, :) = (c_new(:, :).^(-sigma))*prob(:, :)';
    RHS(:, :) = (beta*(1+r-delta)*RHS(:, :)).^(-1/sigma);

    for is = 1:NS
        parfor ia = 1:NA
            c_new(ia, is) = fzero(@(x) foc(x, is, w*eta(is) + (1 + r - delta)*a(ia), RHS, sigma, a_l, a_u, NA), c(ia, is));
        end
    end

    c_new = min(c_new, w*repmat(eta', NA, 1) + (1 + r - delta)*repmat(a, 1, NS) - a_l);
    c_new = max(c_new, 1e-4);

    test = max(abs(c_new - c), [], 'all')./max(abs(c), [], 'all');
    c = 0.2*c_new + 0.8*c;

    aplus = w*repmat(eta', NA, 1) + (1 + r - delta)*repmat(a, 1, NS) - c;

end

end



function foc_v = foc(x_in, is, available, RHS, sigma, a_l, a_u, NA)

% future assets
a_plus = available - x_in;

[ial, iar, varphi] = linint(a_plus, a_l, a_u, NA);
foc_v = varphi*RHS(ial, is) + (1-varphi)*RHS(iar, is);

% get first order condition
foc_v = x_in - foc_v;

end

function obj = valuefunc(a_plus, is, EV, available, beta, sigma, a_l, a_u, NA)

a = linspace(a_l, a_u, NA)';

cons = max(available - a_plus, 1e-10);

[ial, iar, varphi] = linint(a_plus, a_l, a_u, NA);

obj = varphi*EV(ial, is) + (1-varphi)*EV(iar, is);
obj = (cons)^(1-sigma)/(1-sigma) + beta*obj - 100000*abs(cons - available + a_plus);

end

function phi = get_distribution(aplus, a_l, a_u, NA, NS, prob)

a = linspace(a_l, a_u, NA)';

% form transition matrix
% trans is the transition matrix from state at t (row)
% to the state at t+1 (column)
% The eigenvector associated with the unit eigenvalue
% of trans’ is the stationary distribution. %

ial = zeros(NA, NS);
iar = zeros(NA, NS);
varphi = zeros(NA, NS);

for ia = 1:NA
    for is = 1:NS
        [ial(ia, is), iar(ia, is), varphi(ia, is)] = linint(aplus(ia, is), a_l, a_u, NA);
        varphi(ia, is) = max(min(varphi(ia, is), 1), 0);
    end
end

test = 10;
phi = ones(NA, NS)/NA/NS;

while test > 10^(-8)
    phi_new = zeros(NA, NS);
    for ia = 1:NA
        for is = 1:NS
            for is_p = 1:NS
                phi_new(ial(ia, is), is_p) = phi_new(ial(ia, is), is_p) + prob(is, is_p)*varphi(ia, is)*phi(ia, is);
                phi_new(iar(ia, is), is_p) = phi_new(iar(ia, is), is_p) + prob(is, is_p)*(1-varphi(ia, is))*phi(ia, is);
            end
        end
    end
    test=max(abs(phi_new-phi), [], 'all');
    phi = phi_new;
end


end



