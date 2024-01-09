classdef DSM < matlab.mixin.Copyable
% Discrete state model class
% Copyright (c) by Armin Lederer under BSD License
% Last modified: Armin Lederer 03/2020        
    properties
        numstates;
        epsilon_dist;
        sig_eta;
        f;
        h;
    end
    methods
        function obj=DSM(numstates,f,h,epsilon_dist,sig_eta)
            %initialization
            % IN:
            %   numstates       1 x 1   number of states
            %   f               fhandle dynamics
            %   epsilon_dist    1 x 1   process noise distribution
            %   sig_eta         1 x 1   observation noise standard dev.
            % OUT:
            %   obj             DSM     class object
            
            obj.epsilon_dist=epsilon_dist;
            obj.sig_eta=sig_eta;
            obj.numstates=numstates;
            obj.f=f;
            obj.h=h;
        end
        
        function [x,y]=simulate(obj,x)
            %simulate one step of DSM
            % IN:
            %   obj         DSM      class object
            %   x           E x 1    initial state
            % OUT:
            %   x           E x 1    new state
            %   y           D x 1    new observation
            
            epsilon=obj.drawdscrt(obj.epsilon_dist);
            x=mod(x+epsilon-1,obj.numstates)+1;
            
            y=obj.h(x)+random('Normal',0,obj.sig_eta);
        end
        
        function y=output(obj,x)
            %sample observation
            % IN:
            %   obj         DSM      class object
            %   x           E x 1    state
            % OUT:
            %   y           D x 1    observation
            y=obj.h(x)+random('Normal',0,obj.sig_eta);
        end
        
        function x=drawdscrt(obj,p)
            %draw discrete random variable
            % IN:
            %   obj         DSM      class object
            %   p           N x 1    probability distribution
            % OUT:
            %   x           1 x 1    random number
            p=cumsum(p);
            x=rand(1);
            for i=1:length(p)
                if(x<=p(i))
                    x=i-1;
                    break;
                end
            end
        end
    end
end