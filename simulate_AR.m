function shocks = simulate_AR(pi, T)
     
    %##### INPUT/OUTPUT VARIABLES #############################################
     
    % transition matrix
    pi = double(pi);
     
    % should the random seed be initialized at a fixed values
    fixed = false;
     
    %##### OTHER VARIABLES ####################################################
     
%     T = length(shocks);
     
    %##### ROUTINE CODE #######################################################
     
    % assert size equality and get number of simulated shocks
    n = size(pi,1);
    assert(size(pi,2)==n,'tauchen');

    shocks = zeros(T, 1);
     
    % initialize the random seed
    if(fixed)
        rng('default')
    else
        rng('shuffle')
    end
     
    % get first entry
    shocks(1) = floor(n/2)+1;
     
    % now calculate other shocks

            size(pi())
size(pi, 1)

    for j = 2:T
        size(pi, 1)
        shocks(j-1)
        size(pi(shocks(j-1), :))
        shocks(j) = get_tomorrow(pi(shocks(j-1), :));
    end
     
    %##########################################################################
    % Subroutines and functions
    %##########################################################################
    
    function get_tomorrow_v = get_tomorrow(pi)
     
        rand = randn();

        if(size(pi, 1)<2)
                        display('pishock')
            size(pi, 1)
            stop
        end
     
        %##### ROUTINE CODE ###################################################
     
        % get tomorrows value
        for i1 = 1:size(pi, 1)-1
            if(rand <= sum(pi(1:i1), 1))
                get_tomorrow_v = i1;
                return
            end
        end

    
        % else choose last value
        get_tomorrow_v = size(pi, 1)-1;

        if get_tomorrow_v<1
            size(pi, 1)-1
            get_tomorrow_v
            stop
        end
        return
     
    end
    
end
