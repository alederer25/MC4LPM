%% parameter definitions
clear all;
clc;

Ntraj=5;        %number of training trajectories
Nsample=100;    %length of training trajectories

seps=0.0;       %process noise standard deviation
seta=0.05;      %observation noise standard deviation

Nset=100;       %number of trajectories to determine confidence region
Nlength=100;    %length of tractories
sx0=0.5;          %standard deviation of initial state for training

scale=3;        %scaling of the empirical confidence region
delta=5e-2;     %desired maximum violation probability

beta=2;         %variance scaling in uniform error bound

%% generate training data
disp('generating training data');
X=[];
Y=[];
for i=1:Ntraj
    x=sx0*randn([2,1]);
    for j=1:Nsample
        X=[X,x];
        x=sys2d(x)+randn([2,1])*seps;
        y=x+randn([2,1])*seta;
        Y=[Y,y];
    end
end

%% train gp
disp('training Gaussian process');
gp=GPSSM(X,Y,seta);

%% simulate trajectories
disp('simulating random trajectories');

pp=parcluster('local');
if(isempty(gcp('nocreate')))
    parpool(pp.NumWorkers,'IdleTimeout',240);
end



gproll=cell(Nset,1);
X=cell(Nset,1);
Y=cell(Nset,1);

x0=zeros(2,1);
parfor i=1:Nset
    x=x0;
    X{i}=x;
    y=x+seta*randn(2,1);
    Y{i}=y;
    gproll{i}=gp.copy();
    for j=1:Nlength
        [mu,sig]=gproll{i}.predict(x);
        xn=sig.*randn(2,1)+mu;
        x=xn+randn(2,1)*seps;
        X{i}=[X{i},x];
        y=x+seta*randn(2,1);
        Y{i}=[Y{i},y];
        gproll{i}.update(X{i}(:,j),xn);
        
    end
end

%% calculation of mean trajectories
disp('calcluating mean trajectories');

mX=zeros(2,Nlength+1);
mY=zeros(2,Nlength+1);
for i=1:Nset
    mX=mX+real(X{i});
    mY=mY+real(Y{i});
end
mX=mX/Nset;
mY=mY/Nset;

%% computation of maximum confidence sets
disp('determining confidence regions');
rx=zeros(2,Nlength+1);
ry=zeros(2,Nlength+1);
parfor i=1:Nlength+1
    for j=1:Nset
        rx(:,i)=max(rx(:,i),abs(X{j}(:,i)-mX(:,i)));
        ry(:,i)=max(ry(:,i),abs(Y{j}(:,i)-mY(:,i)));
    end
end

rx=scale*rx;
ry=scale*ry;

%% simulate random trajectories to calculate empirical probability
 disp('computing empirical probability')

clear gproll X

Nrollout=ceil((1.1/(sqrt(2)*delta))^3);
gproll=cell(Nrollout,1);
X=cell(Nrollout,1);
Y=cell(Nrollout,1);

pp=parcluster('local');
if(isempty(gcp('nocreate')))
    parpool(pp.NumWorkers,'IdleTimeout',240);
end

x0=zeros(2,1);
parfor i=1:Nrollout
    x=x0;
    X{i}=x;
    y=x+seta*randn(2,1);
    Y{i}=y;
    gproll{i}=gp.copy();
    for j=1:Nlength
        [mu,sig]=gproll{i}.predict(x);
        xn=sig.*randn(2,1)+mu;
        x=xn+randn(2,1)*seps;
        X{i}=[X{i},x];
        y=x+seta*randn(2,1);
        Y{i}=[Y{i},y];
        gproll{i}.update(X{i}(:,j),mu);
    end
end

P=0;
for i=1:Nrollout
    if(all(all(X{i}<=mX+rx))&&all(all(X{i}>=mX-rx)))
        P=P+1;
    end
end
P=P/Nrollout;
disp(['empirical probability ', num2str(P)]);

%% minimization of violation probability
disp('minimizing violation probability');

tbound =@(n,P) -((1 + sqrt(-3))* (8 *n^3* P^3 + 27* n^2* P^2 - 27* n^2* P + 3 *sqrt(3)* sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4* P^3 + ...
     27 *n^4* P^2 - 27 *n^3 *P^3))^(1/3))/(12 *n) + ((1 - sqrt(-3))* (-16 *n^2 *P^2 - 36 *n *P))/(48 *n *(8 *n^3 *P^3 + 27* n^2 *P^2 - 27* n^2 *P ...
     + 3 *sqrt(3) *sqrt(-16 *n^5 *P^4 - 9 *n^4 *P^4 - 54 *n^4 *P^3 + 27 *n^4 *P^2 - 27 *n^3 *P^3))^(1/3)) + P/3;
 
tb = real(tbound(Nrollout,1));
 
ac=@(t) 1-(1-t).*(1-exp(-2*Nrollout*t.^2));
delta=fmincon(ac,1,[],[],[],[],tb,1);
disp(['violation probability ', num2str(delta)]);

%% robust model approach
disp('comparing to uniform error bound approach');

ac=@(x)norm(sys2d(x(1:2))-sys2d(x(3:4)))/(norm(x(1:2)-x(3:4)));
opt=optimset('Display','off');
parfor i=1:1000
    warning('off');
    init=10*(rand(4,1)-0.5);
    [~,Lg(i)]=fmincon(@(x)ac(x),init,[],[],[],[],[],[],[],opt);
end
[~,Lg(1001)]=fmincon(@(x)-ac(x),[0.1;0;-0.1;0],[],[],[],[],[],[],[],opt);
Lg=1.1*max(-Lg);

rrob=zeros(2,1);
Xrob=x0;
x=x0;
for i=1:Nlength
    [x,sig]=gp.predict(x);
    rrob=[rrob,beta*sig+Lg*min(rrob(:,end))];
    Xrob=[Xrob,x];
end

%% plot confidence regions
disp('plotting confidence regions')

ux=(mX+rx)';
lx=(mX-rx)';
conf=[ux;flip(lx,1)];
uxrob=(Xrob+rrob)';
lxrob=(Xrob-rrob)';
confrob(:,1)=max(min([uxrob(:,1);flip(lxrob(:,1),1)],10),-10);
confrob(:,2)=max(min([uxrob(:,2);flip(lxrob(:,2),1)],10),-10);
k=linspace(0,100,101);
klong=[k,flip(k)];

figure(); xlabel('k'); ylabel('x_1');
hold on;
plot(k,mX(1,:),'b')
plot(klong,conf(:,1),'b--')
legend('mean','confidence region bound');
title('trajectory sampling')
figure(); xlabel('k'); ylabel('x_1');
hold on;
plot(k,Xrob(1,:),'b')
plot(klong,confrob(:,1),'b--')
legend('mean','confidence region bound');
title('uniform error bound')

figure(); xlabel('k'); ylabel('x_2');
hold on;
plot(k,mX(2,:),'b')
plot(klong,conf(:,2),'b--')
legend('mean','confidence region bound');
title('trajectory sampling')
figure(); xlabel('k'); ylabel('x_2');
hold on;
plot(k,Xrob(2,:),'b')
plot(klong,confrob(:,2),'b--')
legend('mean','confidence region bound');
title('uniform error bound')



