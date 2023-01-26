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
            [Population,Dec,Mask,FrontNo] = Environmental(Population,Dec,Mask,Problem.N);   %�õ�֮ǰ��
            [actor,critic,obsInfo,actInfo,actor_target,critic_target,Noisemodel]=ddpg(Problem.D);
           % Fitness_pre = calFitness(Population.objs);  %�������и��壬N���������Ӧ��ֵ                   
            %[~,pre_xuhao]=sort(Fitness_pre,'descend');  %pre_xuhaoΪλ��
            %% Optimization 
            juede=1;
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo);
                %if mod(juede,9)==0
                    %[OffDec,OffMask] = Operatorreinfroc(actor,Dec,Mask,Population.decs,Noisemodel,Problem,Population);%������ˣ������Ǵ���pre_xuhao��
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
                if mod(juede,70)==0  %Ŀǰ60���
                    %Pdec=Population.decs;
                    %MatingPool = TournamentSelection(2,Problem.N,FrontNo);
                    Optimum = GetOptimum(Problem,Problem.N);
                    %igd= IGD(Population,Optimum);
                    hv= HV(Population,Optimum);
                    [Dec,actor,critic,actor_target,critic_target,Noisemodel,Population]=reinforc(actor,critic,actor_target,critic_target,obsInfo,actInfo,hv,Dec,Problem,Mask,Noisemodel);
                    PopObj=Population.objs;
                    [FrontNo,~] = NDSort(PopObj,Problem.N);
                    %Ҫ��dec��mask�����������Ȼ����з�֧��������ܺ�operator�����ϣ���Ȼ�൱��û������
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
