function [ial, iar, varphi] = linint(a_plus, a_l, a_u, NA)
% create a grid of values between a_l and a_u
a = linspace(a_l, a_u, NA)';

% find the index of the nearest grid point to a_plus
[idx] = knnsearch(a, a_plus);

% determine the left and right indices and the weighting factor
% for the linear interpolation
if a_plus <= a_l
    % if a_plus is less than or equal to the minimum value of a,
    % set the left index to 1
    ial = 1;
elseif a_plus >= a_u
    % if a_plus is greater than or equal to the maximum value of a,
    % set the left index to the second-to-last index
    ial = NA - 1;
else
    % otherwise, set the left index to the index of the nearest
    % grid point, and set the right index to the next index
    % in the grid
    if a_plus - a(idx) > 0
        ial = idx;
    else
        ial = idx-1;
    end
end
iar = ial + 1;

% calculate the weighting factor for the linear interpolation
varphi = (a(iar) - a_plus)/(a(iar) - a(ial));
end