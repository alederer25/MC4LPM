function [states,outputs]=getconfset(Xtraj,Ytraj,perc)
% get confidence based on random trajectories
% IN:
%   Xtraj           E x N   sample state trajectories
%   Ytraj           D x N   sample output trajectories
%   perc            1 x 1   goal probability
% OUT:
%   states          cells   confidence set
% Copyright (c) by Armin Lederer under BSD License
% Last modified: Armin Lederer 03/2020   

Nlength=size(Xtraj,2);
numstates=max(max(Xtraj));

percx=nthroot(perc,Nlength);

Xperc=Xtraj;
Yperc=Ytraj;

states=cell(Nlength,1);
outputs=cell(Nlength,1);
pupdate=perc;
for i=1:Nlength
    p=0;
    for j=1:numstates
        px(j)=sum(Xperc(:,i)==j)/size(Xperc,1);
    end
    Xnew=[];
    Ynew=[];
    while(p<=percx)
        [~,id]=max(px);
        p=p+px(id);
        px(id)=0;
        states{i}=[states{i},id];
        
        Xnew=[Xnew;Xperc(Xperc(:,i)==id,:)];
        Ynew=[Ynew;Yperc(Xperc(:,i)==id,:)];
        
        outputs{i}=[outputs{i},[min(Yperc(Xperc(:,i)==id,i));max(Yperc(Xperc(:,i)==id,i))]];
    end
    Xperc=Xnew;
    Yperc=Ynew;
    percx=nthroot(pupdate/p,Nlength-i);
    pupdate=pupdate/p;
    
end

end