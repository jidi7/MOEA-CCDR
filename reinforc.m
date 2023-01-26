function [dec_xin,actor,critic,actor_target,critic_target,Noisemodel,Population]=reinforc(actor,critic,actor_target,critic_target,obsInfo,actInfo,hv_qian,dec,Problem,mask_all,Noisemodel)   %
[N,D]=size(dec);
batchSize=2;   %可以就设计的大一点，把population（选择一部分）一个一个的getvalue
%Fitness_after=Fitness;
discountFactor = 0.99;
smoothingfactor=0.01;
numObs = obsInfo.Dimension(1);
numAct = actInfo.Dimension(1);
observationBatch= zeros(numObs,1,batchSize);
actionBatch = zeros(numAct,1,batchSize);
rewardBatch= zeros(1,batchSize);
obs=hv_qian;

% disp(dec)
% pause
%parentdec=dec(MatingPool,:);
%parent1=parentdec(1:batchSize,:);
%parent2=parentdec(N/2+1:N/2+batchSize,:);

%offdec=[parent1;parent2];
%offdec=OperatorGAhalf(offdec);
%Dec=(Dec(1:batchSize,:))';







    for i = 1:batchSize
%         disp(offdec(i,:))
%         pause
         action = getAction(actor,{obs});   
         action = Noisemodel.applyNoise(action);
        [dec_xin,nextobs,reward] = step(obs,action{1},dec,Problem,mask_all,N);   %,isdone
        observationBatch(:,:,i) = obs;
        rewardBatch(:,i) = reward;
        %episodeReward(stepCt) = reward;
        obs = nextobs;
        Population = SOLUTION(dec_xin.*mask_all);
    end
%obs=(Fitness_pre(1:batchSize))';
%更新actionbatch

%offdec=offdec';
%actionBatch(:,:,1:batchSize) = offdec(:,1:end);
%disp(observationBatch)
%disp(rewardBatch)
%pause




%observationBatch(:,:,1:batchSize) = obs;


%reward=step(Fitness_pre(1:batchSize),Fitness_after(1:batchSize)); %这里要改 %step(pre,after,after_xuhao)


%rewardBatch(:,1:batchSize)=reward;
%进行critic的梯度计算，getvalue必须要一个一个的获取了
targetq = zeros(1,batchSize);
yi= zeros(1,batchSize);
for i = 1:batchSize
    if i~=batchSize
        targetq(i) = getValue(critic_target,{observationBatch(:,:,i+1)},{actionBatch(:,:,i+1)});
    else
        targetq(i)=0;
    end
    yi(i)=rewardBatch(i) + discountFactor * targetq(i);
end
%critic的网络更新
lossDatacri.batchSize = batchSize;
lossDatacri.discountedReturn = yi;
criticGradient = gradient(critic,"loss-parameters",...
        [{observationBatch},{actionBatch}],lossDatacri);
critic=optimize(critic,criticGradient);

%actor的网络更新

dQdInput = gradient(critic,'output-input',[{observationBatch},{actionBatch}]);
dQdA = dQdInput{end};
actorGradient = gradient(actor,'output-parameter',{observationBatch},{-dQdA});
actorGradient = rl.internal.dataTransformation.scaleLearnables(actorGradient, 1/batchSize);
actor = optimize(actor, actorGradient);
%更新critic_target
critic_target = syncParameters(critic_target,critic,smoothingfactor);
actor_target = syncParameters(actor_target,actor,smoothingfactor);

%Noisemodel.update();
 function [dec,nextobs,reward] = step(state,dectemp,dec,Problem,mask_all,N)
%MYSTEPFUNCTION 此处显示有关此函数的摘要
%   此处显示详细说明    step(obs,offdec(1,:),dec,Popula,Problem,mask_all);   %,isdone
    %mask=mask';
    %disp(mask)
    %pause
   % disp(dec)
%然后从state中解压状态向量，得到IGD和HV的值
%接着将mask和当前dec相乘得到新的Population，并计算新的IGD和HV值
    %dec_best=Population.best.decs;
    temp=dec;
    %ran=randperm(size(dec_best,1),1);
    %[~,idx] = ismember(dec_best(ran,:),Population.decs,'rows');    %最优向量和实际dec矩阵中哪一行的向量相同，找到最优dec的解的位置索引
    %mask_xin=mask_all;
    %mask_xin(idx,:)=mask;   %mask为网络产生的数，mask中为0的数，mask_xin也为0，否则都在mask_xin取1 
    idx=randperm(N,1);  
    dec(idx,:)=dectemp;
     %disp(idx)
%     pause
    
    
     Populationl = SOLUTION(dec.*mask_all);
%检查新的IGD和HV和之前的相比是否更好
    Optimum = GetOptimum(Problem,Problem.N);
   % igd= IGD(Populationl,Optimum);
     hv= HV(Populationl,Optimum);
%奖励    
% disp(igd)
% disp(igd_qian)
% disp(hv)
% disp(hv_qian)
    if  hv>hv_qian  % if igd<igd_qian && hv>hv_qian
        reward=1;
        nextobs= hv;
    %elseif hv<hv_qian      %elseif igd>igd_qian && hv<hv_qian
       % reward=-5;
    else
        reward=-5;
        nextobs=state;
        dec=temp;
    end
    %disp(reward)
    %pause
    %if reward >0.5
     %   
      %  state(1)  = igd;                          %相当于igd_qian，在这里基本没用
       % state(2) = hv;
        %nextobs=state;
    %else
        %nextobs=state;
       % dec=temp;
   % end
    %disp(dec)
    %pause
end

% function [reward] = step(pre,after)   step(obs,offdec(1,:),dec,Popula,Problem,mask_all);   %,isdone
%     index=after>pre;   %确认fitness是不是越大越好
%     reward(index)=1;   %奖励fitness（after）优于fitness（pre）的
%     reward(~index)=-1;
%     reward=reward';
% end


end
