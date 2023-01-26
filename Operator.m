function [OffDec,OffMask] = Operator(ParentDec,ParentMask,PopDec,Population)  
    [N,D]  = size(ParentMask);
 
    OffMask1=ParentMask(1:N/2,:); 
    OffMask3=ParentMask(N/2+1:end,:); 
    OffMask2=false(N/2,D);   
    OffMask_temp=ParentMask(1:N*4/5,:); 
    Nonzero_mini = any(OffMask_temp,1); 
    allone=all(OffMask1);
    other = Nonzero_mini> allone;   

    OffMask2(:,allone)=OffMask1(:,allone);  %直接学习非零变量的取值（合作）
    OffMask2(:,other) = BinaryCrossover(OffMask1(:,other),OffMask3(:,other));

    dec_best=Population.best.decs;

    one_sum=find(dec_best(1,:)~=0);
    one_length=length(one_sum);
    sparse=one_length/D;
    
    for i=1:N/2  
        flip=find(OffMask2(i,:));
        if length(flip)>D*sparse   
            index=flip(randperm(numel(flip),1));
            OffMask2(i,index)=0;
        elseif length(flip)<D*sparse   
            flip=find(OffMask2(i,:)==0);
            index=flip(randperm(numel(flip),1));
        else
            index=randperm(D,1);     
            OffMask2(i,index)=abs(OffMask2(i,index)-1);    
        end
    end

    OffMask=OffMask2;
    
 %dec的交叉变异   
    Problem = PROBLEM.Current();
    if strcmp(Problem.encoding,'binary')    %如果是二进制变量，offdec全部设置为1
        OffDec = ones(size(OffMask));
    else
       OffDec = OperatorGAhalf(ParentDec);
     end  
     
   
     
     

         %% Remove duplicated solutions     %%删除重复的解
    [~,uni] = unique(OffDec.*OffMask,'rows');
    OffDec  = OffDec(uni,:);
    OffMask = OffMask(uni,:);
    del            = ismember(OffDec.*OffMask,PopDec,'rows');
    OffDec(del,:)  = [];
    OffMask(del,:) = [];




end

