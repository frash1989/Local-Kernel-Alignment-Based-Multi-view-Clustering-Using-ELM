% Unsupervised ELM (US-ELM) for embedding, dimension reduction and clustering.
% Ref: Huang Gao, Song Shiji, Gupta JND, Wu Cheng, Semi-supervised and
% unsupervised extreme learning machines, IEEE Transactions on Cybernetics, 2014

format compact;
clear;
close all;

addpath(genpath('functions'))

% load data
data=load ('iris.txt');
X=data(:,1:end-1);
y=data(:,end);
NC=length(unique(y)); % specify number of clusters

% %%%%%%%%%%%%%%%%% Step 1: construct graph Laplacian %%%%%%%%%%%%%%%%%
% hyper-parameter settings for graph
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;
options.NN=5;

L=laplacian(options,X);

% %%%%%%%%%%%%%%%%% Step 2: Run US-ELM for embedding %%%%%%%%%%%%%%%%%%%
% hyper-parameter settings for us-elm
paras.NE=3; % specify dimensions of embedding
paras.NumHiddenNeuron=2000;
paras.NormalizeInput=0;
paras.NormalizeOutput=0;
paras.Kernel='sigmoid';
paras.lambda=0.1;
elmModel=uselm(X,L,paras);



% %%%%%%%%%%%%%%%%% Step 3: Run k-means for clustering %%%%%%%%%%%%%%%%%
acc_kmeans=[];acc_le=[];acc_uselm=[];
for i=1:100
    [label_kmeans, center] = litekmeans(X,NC,'MaxIter', 200);
    acc_kmeans(i)=accuracy(y,label_kmeans);
    [label_uselm, center] = litekmeans(elmModel.Embed, NC, 'MaxIter', 200);
    acc_uselm(i)=accuracy(y,label_uselm);
end

disp(['Clustering accuracy of k-means, Best: ',num2str(max(acc_kmeans)),...
    ' Average: ',num2str(mean(acc_kmeans))]);
disp(['Clustering accuracy of US-ELM, Best: ',num2str(max(acc_uselm)),...
    ' Average: ',num2str(mean(acc_uselm))]);

% %%%%%%%%%%%%%%%%%%%%% 3-D plot of the results %%%%%%%%%%%%%%%%%%%%%%%
figure(1)
E=X;
hold on
title('The original IRIS data')
view(3)
plot3(E(y==1,1),E(y==1,2),E(y==1,3),'gx','MarkerSize',8,'LineWidth',1.5)
plot3(E(y==2,1),E(y==2,2),E(y==2,3),'c+','MarkerSize',6,'LineWidth',1.5)
plot3(E(y==3,1),E(y==3,2),E(y==3,3),'b.','MarkerSize',10,'LineWidth',1.5)
grid on
axis square


figure(2)
E=elmModel.Embed;
hold on
title('The embedded IRIS data')
view(3)
plot3(E(y==1,1),E(y==1,2),E(y==1,3),'gx','MarkerSize',8,'LineWidth',1.5)
plot3(E(y==2,1),E(y==2,2),E(y==2,3),'c+','MarkerSize',6,'LineWidth',1.5)
plot3(E(y==3,1),E(y==3,2),E(y==3,3),'b.','MarkerSize',10,'LineWidth',1.5)
grid on
axis square
