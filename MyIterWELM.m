%% ELM iterWELM
function [myresult]=MyIterWELM(data,numClust,truth,hidden)
% paramter setting
para.num_hidden_neurons = hidden;
p_array =2.^[-1:1:1];
c_array =10.^[-6:1:6];
num_class = numClust;
myacc_avg =0;
myacc_std =0;
myacc_p=0;
myacc_c=0;
mynmi_avg =0;
mynmi_std =0;
mynmi_p=0;
mynmi_c=0;
mypurity_avg =0;
mypurity_std =0;
mypurity_p=0;
mtpurity_c=0;
for i=1:length(p_array)
    para.p= p_array(i);
    for j=1:length(c_array)
        para.C= c_array(j);
        for trial = 1:50
            %y_init = kmeans(data,num_class);
            [y_init, center] = litekmeans(data,num_class,'MaxIter', 100, 'Replicates',10);
            [~, ite(trial), acc_record{trial}, st{trial}, acc(trial),myacc(trial),mypurity(trial),nmi(trial)] = iterWELM(data, y_init,truth, para);
        end
        NMI(1) = mean(nmi); NMI(2) = std(nmi);
        Acc(1) =mean(myacc);Acc(2) =std(myacc);
        Purity(1) =mean(mypurity);Purity(2)=std(mypurity);
        fprintf('p = %0.4f, C = %.6f \n',para.p, para.C);
        fprintf('ACC: %0.4f(%0.4f)\n', Acc(1), Acc(2));
        fprintf('nmi: %0.4f(%0.4f)\n', NMI(1), NMI(2));
        fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), Purity(2));
        
        if(Acc(1)>myacc_avg)
            myacc_avg =Acc(1);
            myacc_avg = Acc(1);
            myacc_std = Acc(2);
            myacc_p = para.p;
            myacc_c= para.C;
        end
        if(NMI(1)>mynmi_avg)
            mynmi_avg =NMI(1);
            mynmi_avg = NMI(1);
            mynmi_std = NMI(2);
            mynmi_p = para.p;
            mynmi_c= para.C;
        end
        if(Purity(1)>mypurity_avg)
            mypurity_avg =Purity(1);
            mypurity_avg = Purity(1);
            mypurity_std = Purity(2);
            mypurity_p = para.p;
            mypurity_c= para.C;
        end
    end
end
fprintf('Selected Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) log2(p) = %d log c = %d\n', myacc_avg, myacc_std,log2(myacc_p),log10(myacc_c));
fprintf('Best NMI: %0.4f(%0.4f) log2(p) = %d log c = %d\n', mynmi_avg, mynmi_std,log2(mynmi_p),log10(mynmi_c));
fprintf('Best Purity: %0.4f(%0.4f) log2(p) = %d log c = %d\n', mypurity_avg, mypurity_std,log2(mypurity_p),log10(mypurity_c));

