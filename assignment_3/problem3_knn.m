%Classification with k-nn

clear
Data = importdata('test_data_baumarten_group #5.mat');
x = Data.instance_matrix
y = Data.label_vector

y(y==-1) = 0
Mdl = fitcknn(x,y);
d = 0.002; % Step size of the grid
[x1,x2] = meshgrid(min(x(:,1)):d:max(x(:,1)),min(x(:,2)):d:max(x(:,2)));

[~,score] = predict(Mdl,[x1(:),x2(:)]);
%view(Mdl.Trees{1},'Mode','graph')
scoreGrid = reshape(score(:,1),size(x1,1),size(x2,2));

figure
h(1:2) = gscatter(x(:,1),x(:,2),y,'rb');
hold on
contour(x1,x2,scoreGrid);
colorbar;
title('Classification with K-nn')
xlabel('Mean width of Single points')
ylabel('Mean width of First points')
legend('+1: Laubbaum','-1: Nadelbaum')
hold off
%Index = str2double(Index);
[Index,~] = predict(Mdl,x)
cvmdl = crossval(Mdl,'KFold',5)
Loss = kfoldLoss(cvmdl)
ConfMat = confusionmat(y,Index)
TP = ConfMat(1,1)
FP = ConfMat(1,2)
FN = ConfMat(2,1)
TN = ConfMat(2,2)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = (2*Precision*Recall)/(Precision+Recall)
Accuracy = (TP+TN)/(TP+FP+FN+TN)
