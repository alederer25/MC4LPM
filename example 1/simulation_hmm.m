%% parameter definitions
clear all
clc

numstates=5;    %number of states
Nsample=1000;   %number of training samples
Nlength=10;     %length of simulated trajectories
Ntraj=1000;     %number of trajectories to determine confidence region
perc=0.6;       %percentage of trajectories contained in confidence region 
Nrollout=10000; %number of simulated trajectories to determine violation probability

%% generate training data
disp('generating training data');
dsm=DSM(numstates,@(x)mod(x-2,numstates)+1,@(x)2*(x-numstates/2),[0.95,0.05,0.0],0.5);

X=numstates;
Y=dsm.output(X);

for i=2:Nsample
    [X(i),Y(i)]=dsm.simulate(X(i-1));
end


%% train BHMM
disp('training Bayesian Hidden Markov Model');
hmm=BHMM(numstates,Y);

for i=1:1000
    hmm.sample();
end

%% determine confidence region
disp('simulating random trajectories and computing confidence regions');


Xconf=zeros(Ntraj,Nlength);
Yconf=zeros(Ntraj,Nlength);

for j=1:Ntraj
    hmm.sample();
    Xconf(j,1)=numstates;
    Yconf(j,1)=randn(1)*sqrt(hmm.sig2)+hmm.mu(Xconf(j,1));
    for i=2:Nlength
        Xconf(j,i)=hmm.drawdscrt(hmm.A(Xconf(j,i-1),:));
        Yconf(j,i)=randn(1)*sqrt(hmm.sig2)+hmm.mu(Xconf(j,i));
    end
end

[states,outputs] = getconfset(Xconf,Yconf,perc);

mX=mean(Xconf,1);
mY=mean(Yconf,1);



%% calculate violation probability
disp('computing empirical probability')

X=zeros(Nrollout,Nlength);
Y=zeros(Nrollout,Nlength);

for j=1:Nrollout
    hmm.sample();
    X(j,1)=numstates;
    Y(j,1)=randn(1)*sqrt(hmm.sig2)+hmm.mu(X(j,1));
    for i=2:Nlength
        X(j,i)=hmm.drawdscrt(hmm.A(X(j,i-1),:));
        Y(j,i)=randn(1)*sqrt(hmm.sig2)+hmm.mu(X(j,i));
    end
end


for j=1:Nlength
    if(length(states{j})>1)
        pX(:,j)=sum(X(:,j)==states{j},2);
    else
        pX(:,j)=X(:,j)==states{j};
    end
    if(size(outputs{i},2)>1)
        pY(:,j)=sum(Y(:,j)<=outputs{j}(2,:)&Y(:,j)>=outputs{j}(1,:),2);
    else
        pY(:,j)=Y(:,j)<=outputs{j}(2,:)&Y(:,j)>=outputs{j}(1,:);
    end
end
pX=prod(pX,2);
pY=prod(min(pY,1),2);

P=sum(pY.*pX)/Nrollout;
disp(['empirical probability for BHMM ', num2str(P)]);


%% minimize probability
disp('minimizing violation probability');
tbound =@(n,P) -((1 + sqrt(-3))* (8 *n^3* P^3 + 27* n^2* P^2 - 27* n^2* P + 3 *sqrt(3)* sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4* P^3 + ...
     27 *n^4* P^2 - 27 *n^3 *P^3))^(1/3))/(12 *n) + ((1 - sqrt(-3))* (-16 *n^2 *P^2 - 36 *n *P))/(48 *n *(8 *n^3 *P^3 + 27* n^2 *P^2 - 27* n^2 *P ...
     + 3 *sqrt(3) *sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4 *P^3 + 27 *n^4 *P^2 - 27 *n^3 *P^3))^(1/3)) + P/3;
 
tb = real(tbound(Nrollout,1-P));
 
ac=@(t) 1-(P-t).*(1-exp(-2*Nrollout*t.^2));
[topt,delta]=fmincon(ac,1,[],[],[],[],tb,1);
disp(['violation probability for BHMM ', num2str(delta)]);

%% calculate reference mean and confidence interval
disp('calculating confidence region for exact system');

Xconfref=zeros(Ntraj,Nlength);
Yconfref=zeros(Ntraj,Nlength);
for j=1:Ntraj
    Xconfref(j,1)=numstates;
    Yconfref(j,1)=dsm.output(Xconfref(j,1));
    for i=2:Nlength
        [Xconfref(j,i),Yconfref(j,i)]=dsm.simulate(Xconfref(j,i-1));
    end
end

[statesref,outputsref] = getconfset(Xconfref,Yconfref,perc);

mX=mean(Xconfref,1);
mY=mean(Yconfref,1);

%% calculate empirical probability for real system
disp('computing empirical probability for real system')

Xref=zeros(Nrollout,Nlength);
Yref=zeros(Nrollout,Nlength);
for j=1:Nrollout
    Xref(j,1)=numstates;
    Yref(j,1)=dsm.output(Xref(j,1));
    for i=2:Nlength
        [Xref(j,i),Yref(j,i)]=dsm.simulate(Xref(j,i-1));
    end
end

for j=1:Nlength
    if(length(statesref{j})>1)
        pXref(:,j)=sum(Xref(:,j)==statesref{j},2);
    else
        pXref(:,j)=Xref(:,j)==statesref{j};
    end
    if(size(outputsref{j},2)>1)
        pYref(:,j)=sum(Yref(:,j)<=outputsref{j}(2,:)&Yref(:,j)>=outputsref{j}(1,:),2);
    else
        pYref(:,j)=Yref(:,j)<=outputsref{j}(2,:)&Yref(:,j)>=outputsref{j}(1,:);
    end
end
pXref=prod(pXref,2);
pYref=prod(min(pYref,1),2);

Pref=sum(pYref.*pXref)/Nrollout;
disp(['empirical probability for real system ', num2str(Pref)]);

%% minimize violation probability for real system
disp('minimizing violation probability for real system');

tbound =@(n,P) -((1 + sqrt(-3))* (8 *n^3* P^3 + 27* n^2* P^2 - 27* n^2* P + 3 *sqrt(3)* sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4* P^3 + ...
     27 *n^4* P^2 - 27 *n^3 *P^3))^(1/3))/(12 *n) + ((1 - sqrt(-3))* (-16 *n^2 *P^2 - 36 *n *P))/(48 *n *(8 *n^3 *P^3 + 27* n^2 *P^2 - 27* n^2 *P ...
     + 3 *sqrt(3) *sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4 *P^3 + 27 *n^4 *P^2 - 27 *n^3 *P^3))^(1/3)) + P/3;
 
tb = real(tbound(Nrollout,1-Pref));
 
ac=@(t) 1-(Pref-t).*(1-exp(-2*Nrollout*t.^2));
[topt,delta_real]=fmincon(ac,1,[],[],[],[],tb,1);
disp(['violation probability for real system ', num2str(delta_real)]);

