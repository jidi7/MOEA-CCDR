classdef MOERLCC < ALGORITHM
% <multi> <real/binary> <large/none> <constrained/none> <sparse>


%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            if strcmp(Problem.encoding,'binary')
                Dec = ones(Problem.N,Problem.D);
            else
                Dec = unifrnd(repmat(Problem.lower,Problem.N,1),repmat(Problem.upper,Problem.N,1));
            end
            Mask = false(size(Dec));
            for i = 1 : Problem.N
                Mask(i,randperm(end,ceil(rand.^2*end))) = true;
            end
            Population = SOLUTION(Dec.*Mask);
            [Population,Dec,Mask,FrontNo] = Environmental(Population,Dec,Mask,Problem.N);   %用的之前的
            [actor,critic,obsInfo,actInfo,actor_target,critic_target,Noisemodel]=ddpg(Problem.D);
           % Fitness_pre = calFitness(Population.objs);  %计算所有个体，N个个体的适应度值                   
            %[~,pre_xuhao]=sort(Fitness_pre,'descend');  %pre_xuhao为位置
            %% Optimization 
            juede=1;
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo);
                %if mod(juede,9)==0
                    %[OffDec,OffMask] = Operatorreinfroc(actor,Dec,Mask,Population.decs,Noisemodel,Problem,Population);%随机化了，本来是代入pre_xuhao的
                    %disp(OffDec)
                   %pause
                %else
                    %[OffDec,OffMask] = OperatorRandomMask(Dec(MatingPool,:),Mask(MatingPool,:),Population.decs); 
                    [OffDec,OffMask] = Operator(Dec(MatingPool,:),Mask(MatingPool,:),Population.decs,Population);  %Operator(Dec(pre_xuhao,:),Mask(pre_xuhao,:));
                if ~isempty(OffDec)
                    % Offspring        = SOLUTION(OffDec.*OffMask);
                     %[Population,Dec,Mask] = EnvironmentalSelectionEA2([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N);
                        [Population,Dec,Mask,FrontNo] = EnvironmentalSelection([Population,SOLUTION(OffDec.*OffMask)],[Dec;OffDec],[Mask;OffMask],Problem.N);
                        %[Population,Dec,Mask,FrontNo,CrowdDis] = EnvironmentalSelection([Population,SOLUTION(OffDec.*OffMask)],[Dec;OffDec],[Mask;OffMask],Problem.N,0,0);
                end                 
                if mod(juede,70)==0  %目前60最好
                    %Pdec=Population.decs;
                    %MatingPool = TournamentSelection(2,Problem.N,FrontNo);
                    Optimum = GetOptimum(Problem,Problem.N);
                    %igd= IGD(Population,Optimum);
                    hv= HV(Population,Optimum);
                    [Dec,actor,critic,actor_target,critic_target,Noisemodel,Population]=reinforc(actor,critic,actor_target,critic_target,obsInfo,actInfo,hv,Dec,Problem,Mask,Noisemodel);
                    PopObj=Population.objs;
                    [FrontNo,~] = NDSort(PopObj,Problem.N);
                    %要把dec和mask都输出出来，然后进行非支配排序才能和operator连接上，不然相当于没有连接
                end
                 juede=juede+1;
            end
        end
    end
end
function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density

    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end
