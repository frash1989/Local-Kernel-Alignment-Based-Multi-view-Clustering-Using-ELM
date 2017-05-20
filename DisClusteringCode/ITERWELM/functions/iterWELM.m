function [y_s,ite,Acc,obj_func,Acc_end,myacc,mypurity,nmi] = iterWELM(x, y_s, T, para)

para.seed_w = rng;
para.seed_b = rng;

obj_func = zeros(1,50);
Acc = zeros(1,50);
Acc_end = 0;
myacc =0;
mypurity =0;
nmi =0;

[y_m, label] = elm_preprocess(y_s );

for ite = 1:50
    y_s0 = y_s;
    [output,y_m,y_s,weight,OutputWeight] = w_elm( x, y_m, label, para);
    
    obj_func(ite) = 0.5 * para.C * sum(weight.* sum((y_m - output).^2,2)) + 0.5 * norm(OutputWeight,'fro')^2;
    Acc(ite) = accuracy(double(T),y_s);
    if  isequal(y_s0, y_s)
        Acc_end = accuracy(double(T),y_s);
        [result,~] = ClusteringMeasure(T, y_s);
        myacc = result(1);
        mypurity = result(3);
        [~,nmi,~] = compute_nmi (T, y_s);
        break;
    end
end
