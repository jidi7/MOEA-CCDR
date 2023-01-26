classdef MOEACCDR < ALGORITHM
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
            [Population,Dec,Mask,FrontNo] = Environmental(Population,Dec,Mask,Problem.N);   
            [actor,critic,obsInfo,actInfo,actor_target,critic_target,Noisemodel]=ddpg(Problem.D);
            juede=1;
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo);
      
                    [OffDec,OffMask] = Operator(Dec(MatingPool,:),Mask(MatingPool,:),Population.decs,Population);  %Operator(Dec(pre_xuhao,:),Mask(pre_xuhao,:));
                if ~isempty(OffDec)
                   end                 
                if mod(juede,70)==0  
                   
                    Optimum = GetOptimum(Problem,Problem.N);
                 
                    hv= HV(Population,Optimum);
                    [Dec,actor,critic,actor_target,critic_target,Noisemodel,Population]=reinforc(actor,critic,actor_target,critic_target,obsInfo,actInfo,hv,Dec,Problem,Mask,Noisemodel);
                    PopObj=Population.objs;
                    [FrontNo,~] = NDSort(PopObj,Problem.N);
               
                end
                 juede=juede+1;
            end
        end
    end
end

