function [ Output,Y_m,Y_s,weight,output_weight] = w_elm( X, T, label,para)

num_train_data = size(X,1);
num_input_neurons = size(X,2);
number_class = size(label,2);

%%%%%%%%%%% Compute class_size information

class_size = zeros(number_class,1);
for i = 1:number_class
    class_size(i,1) = size(find(T(:,i)==1),1);
end

avg_class_size = mean(class_size);
weight = ones(num_train_data,1);
for i = 1:number_class
    temp = T(:,i)==1;
    weight(temp)= (avg_class_size/class_size(i))^para.p;
end

%%%%%%%%%%% Random generate input weights input_weight and biases bias (b_i) of hidden neurons
rng(para.seed_w);
input_weight = rand(num_input_neurons, para.num_hidden_neurons)*2-1;
rng(para.seed_b);
bias = rand(1,para.num_hidden_neurons);
tempH = X * input_weight;
clear X;                                            
tempH = bsxfun(@plus,tempH,bias);

%%%%%%%%%%% Calculate hidden neuron output matrix H with sig ActFun

H = 1 ./ (1 + exp(-tempH));
clear tempH;                                        

%%%%%%%%%%% Calculate output weights output_weight (beta_i)
if size(H,1) < size(H,2)
    output_weight = H' * ((eye(size(H,1))/para.C + bsxfun(@times, H, weight) * H') \ bsxfun(@times, T, weight));
else
    output_weight = (eye(size(H',1))/para.C + H' * bsxfun(@times, H, weight)) \ (H' * bsxfun(@times, T, weight));
end

%%%%%%%%%%% Make prediction on the training data
Output = H * output_weight; 
Y_s = zeros(size(Output, 1),1);
Y_m = zeros(size(Output, 1),number_class);
for i = 1 : size(Output, 1)
    [~, label_index_expected]=max(Output(i,:));
    Y_s(i,1) = label(1,label_index_expected);
    Y_m(i,label_index_expected)=1;
end
end

