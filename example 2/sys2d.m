function y=sys2d(x)
% inverted pendulum dynamics
% IN:
%   x           E x 1   initial state
% OUT:
%   y           E x 1   next state
% Copyright (c) by Armin Lederer under BSD License
% Last modified: Armin Lederer 03/2020

dt=5e-2;

[t,y]=ode23(@diffeq,[0,dt], x);
y=y(end,:)';

end

function dz=diffeq(t,z)
%differential equation for inverted pendulum
% IN:
%   t           1 x 1   time
%   z           E x 1   state
% OUT:
%   dz          E x 1   time derivative of state

m= 0.15;
g=9.81;
l=0.5;
eta=0.1;

dz=zeros(2,1);
dz(1)=z(2);
dz(2)=g/l*sin(z(1))-eta/(m*l^2)*z(2);

end

