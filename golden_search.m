function [x_bisec, fx_bisec] = golden_search(funcv, x_left_bisec, x_right_bisec)
% GOLDEN_SEARCH - Golden Section Search algorithm for unimodal functions
% Input:
%   - f: the objective function to be minimized
%   - a: the lower bound of the search interval
%   - b: the upper bound of the search interval
%   - tol: the tolerance level for stopping the algorithm
% Output:
%   - xmin: the minimizer of the function f
%   - fmin: the function value at the minimizer
%   - niter: the number of iterations required to reach the minimum

% Golden ratio
coeff = 0.62;

x0 = x_left_bisec;
x3 = x_right_bisec;

% calculate new bisection point and function values
x1 = coeff*x0 + (1-coeff)*x3;
x2 = coeff*x3 + (1-coeff)*x0;

f1 = funcv(x1);
f2 = funcv(x2);

% start iteration process
for iter_bisec = 1:10000

    if abs(x3-x0) <= 1e-3
        if f1 < f2
            fx_bisec = f1;
            x_bisec = x1;
        else
            fx_bisec = f2;
            x_bisec = x2;
        end
        return;
    end

    % calculate new interval
    if f2 < f1
        x0 = x1;
        x1 = x2;
        x2 = coeff*x2 + (1-coeff)*x3;
        f1 = f2;
        f2 = funcv(x2);
    else
        x3 = x2;
        x2 = x1;
        x1 = coeff*x1 + (1-coeff)*x0;
        f2 = f1;
        f1 = funcv(x1);
    end
end

disp('bisection error: no convergence');

end