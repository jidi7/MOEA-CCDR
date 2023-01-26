function [dec_xin,actor,critic,actor_target,critic_target,Noisemodel,Population]=reinforc(actor,critic,actor_target,critic_target,obsInfo,actInfo,hv_qian,dec,Problem,mask_all,Noisemodel)   %
[N,D]=size(dec);
batchSize=2;   %���Ծ���ƵĴ�һ�㣬��population��ѡ��һ���֣�һ��һ����getvalue
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
%����actionbatch

%offdec=offdec';
%actionBatch(:,:,1:batchSize) = offdec(:,1:end);
%disp(observationBatch)
%disp(rewardBatch)
%pause




%observationBatch(:,:,1:batchSize) = obs;


%reward=step(Fitness_pre(1:batchSize),Fitness_after(1:batchSize)); %����Ҫ�� %step(pre,after,after_xuhao)


%rewardBatch(:,1:batchSize)=reward;
%����critic���ݶȼ��㣬getvalue����Ҫһ��һ���Ļ�ȡ��
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
%critic���������
lossDatacri.batchSize = batchSize;
lossDatacri.discountedReturn = yi;
criticGradient = gradient(critic,"loss-parameters",...
        [{observationBatch},{actionBatch}],lossDatacri);
critic=optimize(critic,criticGradient);

%actor���������

dQdInput = gradient(critic,'output-input',[{observationBatch},{actionBatch}]);
dQdA = dQdInput{end};
actorGradient = gradient(actor,'output-parameter',{observationBatch},{-dQdA});
actorGradient = rl.internal.dataTransformation.scaleLearnables(actorGradient, 1/batchSize);
actor = optimize(actor, actorGradient);
%����critic_target
critic_target = syncParameters(critic_target,critic,smoothingfactor);
actor_target = syncParameters(actor_target,actor,smoothingfactor);

%Noisemodel.update();
 function [dec,nextobs,reward] = step(state,dectemp,dec,Problem,mask_all,N)
%MYSTEPFUNCTION �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��    step(obs,offdec(1,:),dec,Popula,Problem,mask_all);   %,isdone
    %mask=mask';
    %disp(mask)
    %pause
   % disp(dec)
%Ȼ���state�н�ѹ״̬�������õ�IGD��HV��ֵ
%���Ž�mask�͵�ǰdec��˵õ��µ�Population���������µ�IGD��HVֵ
    %dec_best=Population.best.decs;
    temp=dec;
    %ran=randperm(size(dec_best,1),1);
    %[~,idx] = ismember(dec_best(ran,:),Population.decs,'rows');    %����������ʵ��dec��������һ�е�������ͬ���ҵ�����dec�Ľ��λ������
    %mask_xin=mask_all;
    %mask_xin(idx,:)=mask;   %maskΪ�������������mask��Ϊ0������mask_xinҲΪ0��������mask_xinȡ1 
    idx=randperm(N,1);  
    dec(idx,:)=dectemp;
     %disp(idx)
%     pause
    
    
     Populationl = SOLUTION(dec.*mask_all);
%����µ�IGD��HV��֮ǰ������Ƿ����
    Optimum = GetOptimum(Problem,Problem.N);
   % igd= IGD(Populationl,Optimum);
     hv= HV(Populationl,Optimum);
%����    
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
      %  state(1)  = igd;                          %�൱��igd_qian�����������û��
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
%     index=after>pre;   %ȷ��fitness�ǲ���Խ��Խ��
%     reward(index)=1;   %����fitness��after������fitness��pre����
%     reward(~index)=-1;
%     reward=reward';
% end


end
