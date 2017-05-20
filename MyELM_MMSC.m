%% ELM MMSC
function  MyELM_MMSC(data,num_class,label,num_views)
for j=1:num_views
    L = j*100;
    [H,OutputWeight]=myelm(data,label,L, 'sigmoid');
    HM{j} = H;
end
nmi_array_avg =[];
ACC_array_avg=[];
Purity_array_avg=[];
nmi_array_std =[];
ACC_array_std=[];
Purity_array_std=[];
%%%%%%%%%%%%%%%%% Step 1: construct graph Laplacian %%%%%%%%%%%%%%%%%
% hyper-parameter settings for graph


p_array =[0.5,1,2];
n_array = [1,5,10];

for pi=1:length(p_array)
    options.LaplacianDegree=p_array(pi);
    for ni=1:length(n_array)
        options.NN=n_array(ni);
        options.GraphWeights='binary';
        options.GraphDistanceFunction='euclidean';
        options.LaplacianNormalize=0;
        L= zeros(size(HM{1},1),size(HM{1},1),num_views);
        for i=1:num_views
            L(:,:,i)=laplacian(options,HM{i});
        end
        
        pr_array= [];
        max_iter = 50;
        step =-2;
        nmi_array_avg =[];
        ACC_array_avg=[];
        Purity_array_avg=[];
        
        nmi_array_std =[];
        ACC_array_std=[];
        Purity_array_std=[];
        pr_array =[];
        while step < 2.0001
            for iter =1:max_iter
                % %%%%%%%%%%%%%%%%% Step 2: Clustering %%%%%%%%%%%%%%%%%
                pr = 10^step;
                [G] = MVSpectralClustering(L, num_class,pr, 'kmeans');
                % %%%%%%%%%%%%%%%%% Step 3: Clustering Evaluation%%%%%%%%%%%%%%%%%
                outlabel = zeros(length(label),1);
                for i= 1:num_class
                    for j =1:length(label)
                        if G(j,i) ==1
                            outlabel(j) = i;%i-1;
                        end
                    end
                end
                %evaluation 1
                [result,~] = ClusteringMeasure(label, outlabel);
                ACCi(iter) =result(1);
                Purityi(iter)=result(3);
                [A nmii(iter) avgenti(iter)] = compute_nmi(label,outlabel);
            end
            nmi(1) = mean(nmii); nmi(2) = std(nmii);
            ACC(1) = mean(ACCi); ACC(2) = std(ACCi);
            Purity(1) =mean(Purityi); Purity(2)=std(Purityi);
            
            nmi_array_avg =[nmi_array_avg;nmi(1)];
            ACC_array_avg=[ACC_array_avg;ACC(1)];
            Purity_array_avg=[Purity_array_avg;Purity(1)];
            
            nmi_array_std =[nmi_array_std;nmi(2)];
            ACC_array_std=[ACC_array_std;ACC(2)];
            Purity_array_std=[Purity_array_std;Purity(2)];
            pr_array =[pr_array;step];
            step =step+0.1;
        end
        fprintf('======================================\n');
        %% Printf the best result
        fprintf('p_degree = %.4f , NN = %d\n',options.LaplacianDegree,options.NN);
        step_ACC = pr_array(find(ACC_array_avg ==max(ACC_array_avg)));
        step_NMI = pr_array(find(nmi_array_avg ==max(nmi_array_avg)));
        step_Purity = pr_array(find(Purity_array_avg ==max(Purity_array_avg)));
        fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
        max_acc = max(ACC_array_avg);
        std_acc = ACC_array_std(find(ACC_array_avg==max(ACC_array_avg)));
        max_nmi = max(nmi_array_avg);
        std_nmi =nmi_array_std(find(nmi_array_avg==max(nmi_array_avg)));
        max_purity =max(Purity_array_avg);
        std_purity =Purity_array_std(find(Purity_array_avg==max(Purity_array_avg)));
        fprintf('ACC=%0.4f(%0.4f),nmi score=%0.4f(%0.4f),Purity=%0.4f(%0.4f)\n',max_acc(1),std_acc(1),max_nmi(1),std_nmi(1),max_purity(1),std_purity(1));
    end
end

% myresult.ACC_avg =max_acc(1);
% myresult.ACC_std =std_acc(1);
% myresult.step_ACC = step_ACC;
% myresult.NMI_avg =max_nmi(1);
% myresult.NMI_std =std_nmi(1);
% myresult.step_NMI = step_NMI;
% myresult.Purity_avg =max_purity(1);
% myresult.Purity_std =std_purity(1);
% myresult.step_Purity = step_Purity;