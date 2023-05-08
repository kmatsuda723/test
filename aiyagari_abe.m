clear all;
close all;
clc;

%
% SMODEL 2
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% set parameter values
%
sigma = 1.50; % risk aversion
beta = 0.98; % subjective discount factor
prob = [ .8 .2; .5 .5]; % prob(i,j) = probability (s(t+1)=sj | s(t) = si)
delta = 0.03; % 1 - depreciation
A = 1.00; % production technology
alpha = 0.25; % capital’s share of income
theta = 0.05; % non-rental income if unemployed is theta*wage
Kstart = 10.0; % initial value for aggregate capital stock
g = 0.20; % iteration の relaxation parameter
%
% form capital grid
%
maxkap = 20; % maximum value of capital grid
inckap = 0.5; % size of capital grid increments
nkap = round(maxkap/inckap+1); % number of grid points
%
% calculate aggregate labor supply
%
[v1,d1]=eig(prob');
[dmax,imax]=max(diag(d1));
probst1=v1(:,imax);
ss=sum(probst1);
probst1=probst1/ss;
pempl=probst1(1,1);
%
N = 1.0*pempl + theta*(1-pempl);
%
% loop to find fixed point for agregate capital stock
%
liter = 1;
maxiter = 50;%iteration に時間がかかるので、

toler = 0.001;% 収束条件に関しては注意すること。あまり厳しくすると収 束しなくなるる
metric = 10;
K = Kstart;
disp('ITERATING ON K');
disp('');
disp(' liter metric meanK Kold');
%
% 均衡総資本を求めるループ
%
while (metric > toler) && (liter <= maxiter)
    %
    % calculate rental rate of capital and wage
    %
    wage = (1-alpha) * A * K^(alpha) * N^(-alpha);% K と N から要素価格を求める
    rent = (alpha) * A * K^(alpha-1) * N^(1-alpha); %
    % tabulate the utility function such that for zero or negative
    % consumption utility remains a large negative number so that
    % such values will never be chosen as utility maximizing
    %
    util1=-10000*ones(nkap,nkap); % utility when employed 
    util2=-10000*ones(nkap,nkap); % utility when unemployed 状態に応じて効用を分離
    for i=1:nkap
        kap=(i-1)*inckap;% 効用関数の作り方は離散近似
        for j=1:nkap
            kapp = (j-1)*inckap;
            cons1 = wage + (rent + 1-delta)*kap - kapp;
            if cons1 > 0
                util1(j,i)=(cons1)^(1-sigma)/(1-sigma);
            end
            cons2 = theta*wage + (rent + 1-delta)*kap - kapp;
            if cons2 > 0
                util2(j,i)=(cons2)^(1-sigma)/(1-sigma);
            end
        end
    end

    %
    % initialize some variables %
    v = zeros(nkap,2);
    decis = zeros(nkap,2);
    test = 10;
    [rs,cs] = size(util1); %
    % iterate on Bellman’s equation and get the decision
    % rules and the value function at the optimum %
    while test ~= 0
        for i=1:cs
            r1(:,i)=util1(:,i)+beta*(prob(1,1)*v(:,1)+ prob(1,2)*v(:,2));
            r2(:,i)=util2(:,i)+beta*(prob(2,1)*v(:,1)+ prob(2,2)*v(:,2));
        end
        [tv1,tdecis1]=max(r1); 
        [tv2,tdecis2]=max(r2);
        tdecis=[tdecis1' tdecis2']; 
        tv=[tv1' tv2'];
        test=max(any(tdecis-decis)); 
        v=tv;
        decis=tdecis;
    end
    decis=(decis-1)*inckap;


    %
    % form transition matrix
    % trans is the transition matrix from state at t (row)
    % to the state at t+1 (column)
    % The eigenvector associated with the unit eigenvalue
    % of trans’ is the stationary distribution. %
    g2=sparse(cs,cs);
    g1=sparse(cs,cs);
    for i=1:cs
        g1(i,tdecis1(i))=1;
        g2(i,tdecis2(i))=1;
    end
    trans=[ prob(1,1)*g1 prob(1,2)*g1; prob(2,1)*g2 prob(2,2)*g2];

    trans=trans';
    probst = (1/(2*nkap))*ones(2*nkap,1);
    test=1;
    while test > 10^(-8)
        probst1 = trans*probst;
        test = max(abs(probst1-probst));
        probst = probst1;
    end
    %
    % vectorize the decision rule to be conformable with probst
    % calculate new aggregate capital stock meanK
    kk=decis(:);
    meanK=probst'*kk;
    %
    % calculate measure over (k,s) pairs
    % lambda has same dimensions as decis %
    lambda=zeros(cs,2);
    lambda(:)=probst; 
    %
    % calculate stationary distribution of k 
    %
    [v1,d1]=eig(prob');
    [dmax,imax]=max(diag(d1));
    probst1=v1(:,imax);
    ss=sum(probst1);
    probst1=probst1/ss;
    probk=sum(lambda'); % stationary distribution of 'captal’ probk=probk’;
    %
    % form metric and update K
    %
    Kold = K;
    Knew = g*meanK + (1-g)*Kold;
    metric = abs((Kold-meanK)/Kold); 
    K = Knew;
    disp([ liter metric meanK Kold ]);
    liter = liter+1;

end
%
% print out results
%
disp('PARAMETER VALUES');
disp('');
disp(' sigma beta delta A alpha theta'); 
disp([ sigma beta delta A alpha theta]); 
disp('');
disp('EQUILIBRIUM RESULTS '); 
disp('');
disp(' K N wage rent');
disp([ Kold N wage rent ]);
%
% simulate life histories of the agent
%
disp('SIMULATING LIFE HISTORY');
k = Kold; % initial level of capital
n = 100; % number of periods to simulate 
s0 = 1; % initial state
hist = zeros(n-1,2);
cons = zeros(n-1,1);
invest = zeros(n-1,1);
grid = [ (0:inckap:maxkap)' ];
[chain,state] = markov_abe(prob,n,s0);
for i = 1:n-1
    hist(i,:) = [ k chain(i) ];
    I1 = round(k/inckap) ;
    I2 = round(k/inckap) + 1;
    if I1 == 0
        I1=1;
        disp('N.B. I1 = 0');
    end
    if I2 > nkap
        I2 = nkap;
        disp('N.B. I2 > nkap');
    end
    weight = (grid(I2,1) - k)/inckap;
    kprime = weight*(decis(I1,chain(i))) + (1-weight)*(decis(I2,chain(i)));

    if chain(i) == 1
        cons(i) = wage + (rent + 1-delta)*k - kprime;
    elseif chain(i) == 2
        cons(i) = wage*theta + (rent + 1-delta)*k - kprime;
    else
        disp('something is wrong with chain');
        chain
    end
    k = kprime; 
    invest(i) = kprime;
end
%
%
subplot(2,2,1),plot((1:n-1)',invest,(1:n-1)',cons);
title('MODEL 2: INVESTMENT AND CONSUMPTION');
% print histmod2
disp('Covariance matrix');
disp([cov(cons,invest)]);
%
% calculate income distribution %
income = [ (rent*grid + wage) (rent*grid + wage*theta) ] ;
[ pinc, index ] = sort(income(:));
plambda = lambda(:);
%
subplot(2,2,2), plot(pinc,plambda(index));
title('MODEL 2: INCOME DISTRIBUTION');
xlabel('INCOME LEVEL');
ylabel('% OF AGENTS');
% print distmod2
%
% calculate capital distribution
%
subplot(2,2,3),plot(grid,probk);
title('MODEL 2: CAPITAL DISTRIBUTION');
xlabel('CAPITAL GRID');
ylabel('% OF AGENTS');
% print capdmod2
%%%%%%%%%%%%%%%%%%%%%%%%

