import numpy as np
from numpy.random import randn
import numba 
from numba import vectorize, float64
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import scipy.stats as st

def simulate_markov(T, pi0, pi):
    """
    Simulates a Markov chain of length T with initial distribution pi0 and transition matrix pi.
    
    Args:
    - T: length of time series
    - pi0: initial distribution over states
    - pi: transition matrix
    
    Returns:
    - states: array of state sequences
    - pi_emp: empirical transition matrix
    """
    # Determine the number of states
    N = pi.shape[0]
    
    # Initialize the state sequence
    states = np.zeros(T, dtype=int)
    
    # Draw the initial state from pi0
    states[0] = np.random.choice(N, p=pi0)
    
    # Generate the state sequence
    for t in range(1, T):
        states[t] = np.random.choice(N, p=pi[states[t-1], :])
    
    # Compute the empirical transition matrix
    pi_emp = np.zeros((N, N))
    for i in range(N):
        mask = (states[:-1] == i) & (states[1:] == np.arange(N)[:, np.newaxis])
        pi_emp[i] = np.sum(mask, axis=1)
        pi_emp[i] /= np.sum(pi_emp[i])
    
    return states

def tauchen(n, mu, rho, sigma):
    # Function to implement Tauchen's method for discretizing a continuous state space
    # Inputs:
    # n: number of grid points
    # mu: mean of the AR(1) process
    # rho: AR(1) coefficient
    # sigma: standard deviation of the error term
    # Outputs:
    # transition_matrix: n x n transition matrix
    # state_space: n x 1 vector of state space points

    m = 1 / np.sqrt(1 - rho**2)

    # Compute the state space
    state_space = np.linspace(mu - m*sigma, mu + m*sigma, n)

    # Compute the distance between grid points
    d = (state_space[n-1] - state_space[0]) / (n - 1)

    # Compute the transition probabilities
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                transition_matrix[i, j] = st.norm.cdf((state_space[0] - rho*state_space[i] + d/2) / sigma)
            elif j == n-1:
                transition_matrix[i, j] = 1 - st.norm.cdf((state_space[n-1] - rho*state_space[i] - d/2) / sigma)
            else:
                z_low = (state_space[j] - rho*state_space[i] - d/2) / sigma
                z_high = (state_space[j] - rho*state_space[i] + d/2) / sigma
                transition_matrix[i, j] = st.norm.cdf(z_high) - st.norm.cdf(z_low)

    return transition_matrix, state_space

sigma = 1.50 # risk aversion
beta = 0.98 # subjective discount factor
delta = 0.03 # depreciation
Z = 1.00 # production technology
alpha = 0.25 # capital’s share of income
Kstart = 10.0 # initial value for aggregate capital stock
g = 0.2 # iteration relaxation parameter
rho = 0.6 # labor productivity persistence
sigma_eps = np.sqrt(0.6*(1-rho**2)) # labor productivity variance




NS = 2
prob, eta = tauchen(NS, -0.7, rho, sigma_eps)
eta = np.exp(eta)

a_l = 0 # minimum value of capital a
a_u = 20 # maximum value of capital a
inckap = 0.05 # size of capital a increments
NA = round((a_u-a_l)/inckap+1) # number of a points
a = np.linspace(a_l, a_u, NA) # grids

d1, v1 = np.linalg.eig(prob.T)
imax = np.argmax(d1)
dmax = d1[imax]
probst1 = v1[:, imax]
ss = np.sum(probst1)
probst1 = probst1 / ss # ss distribution
HH = np.sum(eta*probst1) # aggregate effective labor



Nsim = 1
Tsim = 50
is_t = np.zeros((Nsim, Tsim), dtype=int)
for i in range(Nsim):
    is_t[i, :] = simulate_markov(Tsim, probst1, prob)

trans = 0.1
taul = trans/HH

liter = 1
itermax = 50
toler = 0.001 # warning: this doens't converge if tolerance is too small
metric = 100 # initial difference
KK = Kstart # initial capital

def solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans):
    # Create a grid of asset holdings
    a = np.linspace(a_l, a_u, NA)

    # Initialize the utility function to a large negative number for zero or negative consumption
    util = np.full((NA, NA, NS), -10000)

    # Calculate utility for each asset combination and shock
    for i in range(NA):
        kap = a[i]
        for j in range(NA):
            kapp = a[j]
            for is_ in range(NS):
                cons = (1-taul)*w*eta[is_] + trans + (1 + r - delta)*kap - kapp
                if cons > 0:
                    util[j, i, is_] = cons**(1-sigma)/(1-sigma)

    # Initialize some variables
    v = np.zeros((NA, NS))
    aplus = np.zeros((NA, NS))
    tv_help = np.zeros((NS, NA))
    tdecis_help = np.zeros((NS, NA), dtype=int)
    tv = np.zeros((NA, NS))
    tdecis = np.full((NA, NS), -10000)
    iaplus = np.full((NA, NS), -10000)

    test = 10
    rs, cs, _ = util.shape
    r = np.zeros((NA, NA, NS))

    # Iterate on Bellman's equation and get the decision rules and the value function at the optimum
    while test != 0:
        for i in range(cs):
            for is_ in range(NS):
                r[:,i,is_] = util[:,i,is_] + beta*np.dot(v[:,:], prob[is_,:].T)

        for i in range(cs):
            for is_ in range(NS):
                tv_help[is_, i] = np.max(r[:, i, is_])
                tdecis_help[is_, i] = np.argmax(r[:, i, is_])
            
        tv = tv_help.T
        tdecis = tdecis_help.T

        test = np.max(tdecis - iaplus)
        v = tv
        iaplus = tdecis.copy()

    aplus = a[iaplus]
    return aplus

def linint(a_plus, a_l, a_u, NA):
    # create a grid of values between a_l and a_u
    a = np.linspace(a_l, a_u, NA)

    # find the index of the nearest grid point to a_plus
    idx = np.argmin(np.abs(a - a_plus))

    # determine the left and right indices and the weighting factor
    # for the linear interpolation
    if a_plus <= a_l:
        # if a_plus is less than or equal to the minimum value of a,
        # set the left index to 0
        ial = 0
    elif a_plus >= a_u:
        # if a_plus is greater than or equal to the maximum value of a,
        # set the left index to the second-to-last index
        ial = NA - 2
    else:
        # otherwise, set the left index to the index of the nearest
        # grid point, and set the right index to the next index
        # in the grid
        if a_plus - a[idx] > 0:
            ial = idx
        else:
            ial = idx - 1
    iar = ial + 1

    # calculate the weighting factor for the linear interpolation
    varphi = (a[iar] - a_plus) / (a[iar] - a[ial])

    return ial, iar, varphi

# def linint(x, x1, x2, n):
#     if x <= x1:
#         return 1, 1, 1
#     if x >= x2:
#         return n, n, 1
#     phi = (x - x1) / (x2 - x1) * (n - 1)
#     il = int(np.floor(phi) + 1)
#     ir = int(np.ceil(phi) + 1)
#     varphi = phi - np.floor(phi)
#     return il, ir, varphi

def get_distribution(aplus, a_l, a_u, NA, NS, prob):
    # Generate a linearly spaced vector of NA values between a_l and a_u
    a = np.linspace(a_l, a_u, NA)
    
    # Initialize the indices of aplus values rounded down, rounded up, and the blending coefficients
    ial = np.zeros((NA, NS), dtype=int)
    iar = np.zeros((NA, NS), dtype=int)
    varphi = np.zeros((NA, NS))
    
    # Loop over all values of a and s to interpolate aplus values and calculate blending coefficients
    for ia in range(NA):
        for is_ in range(NS):
            ial[ia, is_], iar[ia, is_], varphi[ia, is_] = linint(aplus[ia, is_], a_l, a_u, NA)
            varphi[ia, is_] = max(min(varphi[ia, is_], 1), 0) # Clip varphi values to be within [0, 1]
    
    test = 10 # Initialize a test value to be greater than 10^-8
    phi = np.ones((NA, NS)) / NA / NS # Initialize the distribution phi to be uniform
    
    # Loop until the test value is less than 10^-8
    while test > 1e-8:
        phi_new = np.zeros((NA, NS)) # Initialize a new distribution phi_new to be all zeros
        # Loop over all values of a, s, and s'
        for ia in range(NA):
            for is_ in range(NS):
                for is_p in range(NS):
                    # Update phi_new using the interpolation indices, blending coefficients, and probabilities
                    phi_new[ial[ia, is_], is_p] += prob[is_, is_p] * varphi[ia, is_] * phi[ia, is_]
                    phi_new[iar[ia, is_], is_p] += prob[is_, is_p] * (1 - varphi[ia, is_]) * phi[ia, is_]
        test = np.max(np.abs(phi_new - phi)) # Calculate the maximum difference between phi_new and phi
        phi = phi_new # Update phi to be phi_new
    
    return phi

def valuefunc(a_plus, is_, EV, available, beta, sigma, a_l, a_u, NA):
    # Computes the value function for a given interest rate state (is) and
    # choice of additional assets (a_plus) given the current state variables.
    # The value function is defined as the sum of the immediate utility (cons)
    # and the discounted expected value of future utility (EV), minus a large
    # penalty term for infeasible choices of a_plus.
    #
    # Inputs:
    # a_plus: vector of additional assets to choose from
    # is: current interest rate state
    # EV: expected value function for the next period
    # available: current amount of available assets
    # beta: discount factor
    # sigma: coefficient of relative risk aversion
    # a_l, a_u: lower and upper bounds for the grid of asset values
    # NA: number of points on the grid of asset values
    #
    # Output:
    # obj: value function for the given interest rate state and choice of
    # additional assets.

    a = np.linspace(a_l, a_u, NA) # Generate evenly spaced grid of asset values

    cons = np.maximum(available - a_plus, 1e-10) # Compute consumption

    # Compute the interpolated indices and weights for linear interpolation
    # of the expected value function
    ial, iar, varphi = linint(a_plus, a_l, a_u, NA)

    # Compute the objective function as the sum of immediate utility, the
    # discounted expected value of future utility, and a large penalty term
    # for infeasible choices of a_plus.
    obj = varphi*EV[ial, is_] + (1-varphi)*EV[iar, is_]
    obj = (cons)**(1-sigma)/(1-sigma) + beta*obj - 100000*np.abs(cons - available + a_plus)

    return obj

from scipy.optimize import minimize_scalar


def solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans):
    # Grid of asset holdings
    a = np.linspace(a_l, a_u, NA)

    # Initialize variables
    EV = -82 * np.ones((NA, NS))
    EV_new = np.zeros((NA, NS))
    v = np.zeros((NA, NS))
    aplus = np.zeros((NA, NS))
    test = 10

    # Iterate on Bellman's equation and get the decision
    # rules and the value function at the optimum
    while test > 10**(-4):
        for is_ in range(NS):
            for ia in range(NA):
                # use minimize_scalar to solve for optimal asset holding
                def negative_valuefunc(x):
                    return -valuefunc(x, is_, EV, (1 - taul) * w * eta[is_] + trans + (1 + r - delta) * a[ia], beta, sigma, a_l, a_u, NA)



                result = minimize_scalar(negative_valuefunc, bounds=(a_l, a_u), method='bounded')



                aplus[ia, is_] = result.x
                v[ia, is_] = -result.fun

        for is_ in range(NS):
            # calculate the new value function by taking the expectation of the
            # value function at the optimal asset level for each state tomorrow
            EV_new[:, is_] = np.dot(v[:,:], prob[is_,:].T)

        # calculate the test statistic for convergence
        test = np.max(np.abs(EV_new - EV))

        print(aplus)
        # print(EV)
        # print(EV_new)
        print(test)

        EV = EV_new  # update value function

    return aplus

from scipy.optimize import root_scalar

def solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans):

    aplus = np.zeros((NA, NS))

    # Grid of asset holdings
    a = np.linspace(a_l, a_u, NA)

    # Initialize variables
    RHS = np.zeros((NA, NS))
    c = np.zeros((NA, NS))

    # Compute current consumption
    for is_ in range(NS):
        c[:, is_] = a + eta[is_]
    c_new = c.copy()

    # Set initial test value for convergence
    test = 10

    # Loop until convergence is reached
    while test > 1e-4:
        # Compute the RHS of the Euler equation
        RHS[:, :] = (c_new[:, :] ** (-sigma)) @ prob[:, :].T
        RHS[:, :] = (beta * (1 + r - delta) * RHS[:, :]) ** (-1 / sigma)

        # Solve for optimal consumption using fzero
        for is_ in range(NS):
            for ia in range(NA):
                # Solve the first-order condition for consumption
                x0 = c[ia, is_]
                available = (1-taul)*w*eta[is_] + trans + (1 + r - delta)*a[ia]
                foc_fun = lambda x: foc(x, is_, available, RHS, sigma, a_l, a_u, NA)

                sol = root_scalar(foc_fun, bracket=[a_l, a_u])
                c_new[ia, is_] = sol.root
                c_new[ia, is_] = np.minimum(c_new[ia, is_], (1 - taul) * w * eta[is_] + trans + (1 + r - delta) * a[ia] - a_l)
                aplus[ia, is_] = w * eta[is_] + (1 + r - delta) * a[ia] - c_new[ia, is_]

        # Implement constraints
        # c_new = np.minimum(c_new, (1 - taul) * w * np.tile(eta, (NA, 1)).T + trans + (1 + r - delta) * np.tile(a, (NS, 1)).T - a_l)
        # c_new = np.minimum(c_new, (1 - taul) * w * np.tile(eta.T, (NA, 1)) + trans + (1 + r - delta) * np.tile(a.T, (NS, 1)) - a_l)
        # c_new = np.minimum(c_new, (1 - taul) * w * np.tile(eta, (NA, 1)) + trans + (1 + r - delta) * np.tile(a.T, (NA, 1)).T - a_l)

        c_new = np.maximum(c_new, 1e-4)

        # Calculate convergence criterion
        test = np.max(np.abs(c_new - c)) / np.max(np.abs(c))

        # Update c using a dampened update rule
        c = 0.2 * c_new + 0.8 * c

    # Calculate optimal future assets
    # aplus = w * np.tile(eta, (NA, 1)).T + (1 + r - delta) * np.tile(a, (NS, 1)).T - c

    return aplus

def foc(x_in, is_, available, RHS, sigma, a_l, a_u, NA):
    # future assets
    a_plus = available - x_in

    # linear interpolation
    ial, iar, varphi = linint(a_plus, a_l, a_u, NA)
    foc_v = varphi * RHS[ial, is_] + (1 - varphi) * RHS[iar, is_]

    # get first order condition
    foc_v = x_in - foc_v

    return foc_v





while (metric > toler) and (liter <= itermax):

    # calculate rental rate of capital and w

    w = (1-alpha) * Z * KK**(alpha) * HH**(-alpha)
    r = (alpha) * Z * KK**(alpha-1) * HH**(1-alpha)



    ####################################################
    # Solving for households optimization (policy function of assets)
    # choose one of them
    # 1. grid search
    # 2. bisection minimization
    # 3. first order condition
    ####################################################

    # aplus = solve_household_gs(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)
    # aplus = solve_household_interp(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)
    aplus = solve_household_foc(a_l, a_u, r, w, prob, delta, beta, sigma, eta, NA, NS, taul, trans)

    ####################################################
    # Loop for finding eq distribution and capital
    ####################################################

    # eq distribution

    phi = get_distribution(aplus, a_l, a_u, NA, NS, prob)

    # new aggregate capital

    meanK = np.sum(phi*aplus)

    # capital holdings distribution

    probk = np.sum(phi, axis=1)

    ####################################################
    # Loop for finding eq capital with simulations
    ####################################################

    # a_t = np.zeros((Nsim, Tsim+1))
    # aplus_t = np.zeros((Nsim, Tsim+1))
    # c_t = np.zeros((Nsim, Tsim))

    # a_t[:, 0] = KK
    # for i in range(Nsim):
    #     for it in range(Tsim):
    #         ial, iar, varphi = linint(a_t[i, it], a_l, a_u, NA)
    #         a_t[i, it+1] = varphi*aplus[ial, is_t[i, it]] + (1-varphi)*aplus[iar, is_t[i, it]]
    #         aplus_t[i, it] = a_t[i, it+1]
    #         c_t[i, it] = w*eta[is_t[i, it]] + (1 + r - delta)*a_t[i, it] - aplus_t[i, it]
    #
    # meanK = np.sum(a_t[:, Tsim])/Nsim

    ####################################################
    # Loop for finding eq capital
    ####################################################

    # form metric and update KK

    Kold = KK
    Knew = g*meanK + (1-g)*Kold
    metric = abs((Kold-meanK)/Kold)

    KK = Knew
    print([ liter, metric, meanK, Kold])
    liter += 1

print('PARAMETER VALUES')
print('')
print(' sigma beta delta Z alpha')
print([sigma, beta, delta, Z, alpha])
print('')
print('EQUILIBRIUM RESULTS ')
print('')
print(' KK HH w r')
print([KK, HH, w, r])

print('SIMULATING LIFE HISTORY')


# initial
a_t = np.zeros((Nsim, Tsim+1))
aplus_t = np.zeros((Nsim, Tsim+1))
c_t = np.zeros((Nsim, Tsim))
a_t[:, 0] = KK

# simulation
for i in range(Nsim): # agents
    for it in range(Tsim): # time
        ial, iar, varphi = linint(a_t[i, it], a_l, a_u, NA)
        a_t[i, it+1] = varphi*aplus[ial, is_t[i, it]] + (1-varphi)*aplus[iar, is_t[i, it]]
        aplus_t[i, it] = a_t[i, it+1]
        c_t[i, it] = (1-taul)*w*eta[is_t[i, it]] + trans + (1 + r - delta)*a_t[i, it] - aplus_t[i, it]

# plots
plt.subplot(2, 2, 1)
plt.plot(range(1, Tsim+1), a_t[0, 1:Tsim+1]-a_t[0, :Tsim], range(1, Tsim+1), c_t[0, :])
plt.title('MODEL 2: INVESTMENT AND CONSUMPTION')
print('Covariance matrix:')
print(np.cov(c_t[0, :], a_t[0, 1:Tsim+1]-a_t[0, :Tsim]))
#
# calculate income distribution %
income = np.array([(r*a + w*eta[0]), (r*a + w*eta[1])])
income = income.T
pinc, index = np.sort(income.flatten('F')), np.argsort(income.flatten('F'))
plambda = phi.flatten('F')
#
plt.subplot(2, 2, 2)
plt.plot(pinc, plambda[index])
plt.title('MODEL 2: INCOME DISTRIBUTION')
plt.xlabel('INCOME LEVEL')
plt.ylabel('% OF AGENTS')
#
# calculate capital distribution
#
plt.subplot(2, 2, 3)
plt.plot(a, probk)
plt.title('MODEL 2: CAPITAL DISTRIBUTION')
plt.xlabel('CAPITAL a')
plt.ylabel('% OF AGENTS')