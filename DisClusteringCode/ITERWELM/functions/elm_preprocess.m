function [ T, label] = elm_preprocess( T )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
NumberofTrainingData=size(T,1);
%%%%%%%%%%%% Preprocessing the data of classification
label = unique(T)';
number_class=size(label,2);

%%%%%%%%%% Processing the targets of training
temp_T=zeros(NumberofTrainingData,number_class);
for i = 1:NumberofTrainingData
    temp_T(i,label == T(i,1))= 1;
end
T = temp_T;
end

