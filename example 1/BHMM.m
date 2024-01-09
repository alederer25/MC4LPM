classdef BHMM < matlab.mixin.Copyable
% Bayesian Hidden Markov Model class
% Copyright (c) by Armin Lederer under BSD License
% Last modified: Armin Lederer 03/2020        
    properties
        numstates;
        range;
        
        mu;
        sig2;
        initial;
        X;
        A;
        beta;
        
        R;
        xi;
        kappa;
        alpha;
        g;
        h;
        
        fsample;
        
        Y;
    end
    methods
        function obj=BHMM(numstates,Y)
            %initialization
            % IN:
            %   numstates   1 x 1   number of hidden states
            %   Y           E x N   output training data
            % OUT:
            %   obj         BHMM    class object
            
            obj.numstates=numstates;
            obj.Y=Y;
            
            obj.R=max(Y)-min(Y);
            obj.xi=(max(Y)+min(Y))/2;
            obj.kappa=1/obj.R^2;
            obj.alpha=2;
            obj.g=0.2;
            obj.h=10/obj.R^2;
            
            obj.beta=obj.g/obj.h;
            for i=1:numstates
                obj.mu(i)=min(Y)+obj.R/(2*numstates)+(i-1)*obj.R/numstates;
            end
            for i=1:length(Y)
                for j=1:numstates
                    dist(j)=(obj.mu(j)-Y(i))^2;
                end
                [~,obj.X(i)]=min(dist);
            end
            obj.sig2=sum((Y-obj.mu(obj.X(i))).^2)/length(Y);
            for i=1:obj.numstates
                id=find(obj.X==i);
                for j=1:obj.numstates
                    if(id(end)==length(obj.X))
                        id(end)=[];
                    end
                    n(i,j)=sum(obj.X(id+1)==j);
                end
                obj.A(i,:)=n(i,:)/length(id);
            end
            obj.initial=1/numstates*ones(numstates,1);
        end
        
        function sample(obj)
            %draw random hmm from posterior distribution
            % IN:
            %   obj         BHMM     class object
            
            %update mu
            for i=1:obj.numstates
                S=sum(obj.Y(obj.X==i));
                n=sum(obj.X==i);
                obj.mu(i)=randn(1)*sqrt(obj.sig2/(n+obj.kappa*obj.sig2))+(S+obj.kappa*obj.xi*obj.sig2)/(n+obj.kappa*obj.sig2);
            end
            
            %update sig2
            obj.sig2=1/gamrnd(obj.alpha+0.5*length(obj.Y), 1/(obj.beta+0.5*sum((obj.Y-obj.mu(obj.X)).^2)));
            
            %update beta
            obj.beta=random('Gamma',obj.g+obj.alpha,1/(obj.h+1/obj.sig2));
            
            %update A
            n=zeros(obj.numstates,obj.numstates);
            for i=1:obj.numstates
                id=find(obj.X==i);
                if(~isempty(id))
                    for j=1:obj.numstates
                        if(id(end)==length(obj.X))
                            id(end)=[];
                        end
                        n(i,j)=sum(obj.X(id+1)==j);
                    end
                else
                    n(i,:)=zeros(1,obj.numstates);
                end
                obj.A(i,:)=obj.drawdirichlet(n(i,:)+1,1);
            end
            
            obj.initial=zeros(1,obj.numstates);
            obj.initial(obj.numstates)=1;
            
            obj.X(1)=obj.numstates;
                
            for i=2:length(obj.X)-1
                for j=1:obj.numstates
                    tab(j)=obj.A(obj.X(i-1),j)*normpdf(obj.Y(i),obj.mu(j),obj.sig2)*obj.A(j,obj.X(i));%*ptheta;
                end
                obj.X(i)=obj.drawdscrt(tab/sum(tab));
            end
            for j=1:obj.numstates
                tab(j)=obj.A(obj.X(end-1),j)*normpdf(obj.Y(end),obj.mu(j),obj.sig2);
            end
            obj.X(end)=obj.drawdscrt(tab/sum(tab));
            
        end
    end
    
    methods(Static)
        function x = drawdirichlet(shape,num)
            %draw dirichlet random variable
            % IN:
            %   shape       N x 1    shape parameters of the Gamma dist.
            %   num         1 x 1    number of random variables
            % OUT:
            %   x           num x 1  random numbers

            scale = 1;
            l = length(shape);
            shape = repmat(shape,num,1);
            x = gamrnd(shape,scale,[num,l]);
            x = x ./ repmat(sum(x,2),1,l);
        end
        
        function x=drawdscrt(p)
            %draw discrete random variable
            % IN:
            %   p           N x 1    probability distribution
            % OUT:
            %   x           1 x 1    random number
            p=cumsum(p);
            x=rand(1);
            for i=1:length(p)
                if(x<=p(i))
                    x=i;
                    break;
                end
            end
        end
    end
end