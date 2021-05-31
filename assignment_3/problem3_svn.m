clear
Data = importdata('test_data_baumarten_group #5.mat');
Data1 = importdata('test_data_baumarten_group #1.mat');
x = Data.instance_matrix;
y = Data.label_vector;

% SVMModel = fitcsvm(x,y, 'KernelFunction','rbf', 'BoxConstraint', Inf, 'ClassNames', [-1,1], 'OutlierFraction', 0.05)
% SVMModel = fitcsvm(x,y, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'ClassNames', [-1,1], 'OutlierFraction', 0.05)
% SVMModel = fitcsvm(x,y, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'KernelScale', 'auto', 'ClassNames', [-1,1], 'OutlierFraction', 0.05);
SVMModel = fitcsvm(x,y, 'Standardize', true, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'ClassNames', [-1,1]);
% Standardize: Standardize the prediction
% KernelFunction: Radial Basis Kernel
% BoxContraint: infinity -> sehr weich
% SVMModel = fitcsvm(x,y, 'KernelFunction', 'RBF', 'BoxConstraint', Inf, 'ClassNames', [-1,1]);

svInd = SVMModel.IsSupportVector;

d = 0.002; % Step size of the grid

[x1,x2] = meshgrid(min(x(:,1)):d:max(x(:,1)), min(x(:,2)):d:max(x(:,2)));
xGrid = [x1(:), x2(:)];
Index = predict(SVMModel,x);

[~,score] = predict(SVMModel, [x1(:), x2(:)]);
scoreGrid = reshape(score(:, 1), size(x1, 1), size(x2, 2));

figure
h(1:2) = gscatter(x(:, 1), x(:, 2), y, 'rb');
hold on
plot(x(svInd, 1), x(svInd, 2), 'ro', 'MarkerSize', 10);
contour(x1, x2, scoreGrid);
colorbar;
title('Outlier Detection via two class SVM');
xlabel('Mean width of Single points');
ylabel('Mean width of First points');
legend('+1: Laubbaum','-1: Nadelbaum','Support Vectors');

hold off
cvmdl = crossval(SVMModel, 'KFold', 5);
Loss = kfoldLoss(cvmdl)

ConfMat = confusionmat(y, Index)

TP = ConfMat(1,1);
FP = ConfMat(1,2);
FN = ConfMat(2,1);
TN = ConfMat(2,2);

Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
F1_Score = (2*Precision*Recall)/(Precision+Recall)
Accuracy = (TP+TN)/(TP+FP+FN+TN)

