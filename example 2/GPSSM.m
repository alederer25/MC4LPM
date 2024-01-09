classdef GPSSM < matlab.mixin.Copyable
% Gaussian process state space model class
% Copyright (c) by Armin Lederer under BSD License
% Last modified: Armin Lederer 03/2020    
    properties
        X;
        Y;
        hyp;
        kernel;
        dx;
        invK;
        num;
        alpha;
    end
    methods
        function obj=GPSSM(X,Y,sn,varargin)
            %initialization
            % IN:
            %   X           1 x 1   input training data
            %   Y           E x N   output training data
            %   sn          1 x 1   noise variance
            %   varargin    struct  name-value pairs of parameters
            % OUT:
            %   obj         GPSSM   class object
            
            hyp=cell(size(Y,1),1);
            K=cell(size(Y,1),1);
            alpha=zeros(size(Y,1),size(Y,2));
            
            for i=1:size(Y,1)
                if(nargin>3)
                    hyp{i}=varargin{i};
                else
                    gp=fitrgp(X',Y(i,:),'FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential','Standardize',0,'ConstantSigma',true,'Sigma',sn);
                    hyp{i}.l=gp.KernelInformation.KernelParameters(1:end-1);
                    hyp{i}.sf=gp.KernelInformation.KernelParameters(end);
                    hyp{i}.sn=gp.Sigma;
                end
                
                K{i}=obj.kern(X,X,hyp{i});
                K{i}=chol(K{i}+hyp{i}.sn^2*eye(size(K{i},1)));
                alpha(i,:)=(K{i}\(K{i}'\Y(i,:)'))';
                K{i}= K{i} \ eye(size(K{i})) / K{i}';
            end
            
            obj.kernel=@(x1,x2,i)hyp{i}.sf^2*exp(-0.5*(x1-x2)'*diag(1./hyp{i}.l.^2)*(x1-x2));
            obj.hyp=hyp;
            obj.dx=size(Y,1);
            obj.invK=K;
            obj.num=size(X,2);
            obj.X=X;
            obj.Y=Y;
            obj.alpha=alpha;
        end
        
        function K=kern(obj,X1,X2,hyp)
            %Kernel matrix computation
            % IN:
            %   obj         GPSSM    class object
            %   X1          E x N1   input data 1
            %   X2          E x N2   input data 2
            %   hyp         struct   GP Hyperparameters
            % OUT:
            %   K           N1 x N2  covariance matrix

            kernel=@(x1,x2)hyp.sf^2*exp(-0.5*sum((x1-x2).^2./hyp.l.^2),1);
            
            K=zeros(size(X1,2),size(X2,2));
            for j=1:size(X2,2)
                K(:,j)=kernel(X1,X2(:,j));
            end
        end
        
        
        function [mu,sig]=predict(obj,x)
            %GP prediction
            % IN:
            %   obj         GPSSM    class object
            %   x           E x 1    test point
            % OUT:
            %   mu          E x 1    predictive mean
            %   sig         E x 1    predictive standard deviation
            
            mu=zeros(size(obj.alpha,1),1);
            sig=mu;
            for i=1:size(obj.alpha,1)
                k=obj.kern(obj.X,x,obj.hyp{i});
                mu(i)=obj.alpha(i,:)*k;
                sig(i)=real(sqrt(obj.hyp{i}.sf^2-k'*obj.invK{i}*k));
            end
            if(~prod(isreal(mu)))
                dummy;
            end
            
        end
        
        function update(obj, x, y)
            %update of GP distribution
            % IN:
            %   obj         GPSSM    class object
            %   x           E x 1    new input data
            %   y           E x 1    new output data
            
            obj.alpha=zeros(obj.dx,size(obj.alpha,2)+1);
            for i=1:obj.dx
                obj.invK{i} = obj.updateInv(obj.kern(obj.X,x,obj.hyp{i}), obj.kernel(x,x,i)+1e-4,i);%noise variance is necessary for regularization to ensure numerical stability
                obj.alpha(i,:) = obj.invK{i} * [obj.Y(i,:) y(i)]';
            end
            obj.X = [obj.X x];
            obj.Y = [obj.Y y];
            
        end
        
        function inv = updateInv(obj, b, c, i)
            %rank one update of the inverse covariance matrix
            % IN:
            %   obj         GPSSM    class object
            %   b           N x 1    covariance to existing data
            %   c           1 x 1    prior variance of new point
            %   i           1 x 1    matrix index
            % OUT:
            %   inv          N x N   new inverse matrix           
            
            k = c - b'*obj.invK{i}*b;
            d = obj.invK{i}*b;
            inv = [obj.invK{i} + d*d'/k  -d/k;...
                -d'/k                  1/k];
            
        end
    end
end




