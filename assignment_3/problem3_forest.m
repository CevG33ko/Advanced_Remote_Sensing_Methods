clear
Data = importdata('test_data_baumarten_group #5.mat');
x = Data.instance_matrix;
y = Data.label_vector;
%y = ones(size(x, 1), 1)
y(y==-1) = 0;
Mdl = TreeBagger(20, x, y, 'Method', 'classification',  'Surrogate', 'on', 'OOBPredictorImportance', 'on',  "OOBPrediction", "on");
d = 0.005; % Step size of the grid
[x1, x2] = meshgrid(min(x(:, 1)):d:max(x(:, 1)), ...
min(x(:, 2)):d:max(x(:, 2)));
xGrid = [x1(:), x2(:)]
[~, score] = predict(Mdl, [x1(:), x2(:)]);
[Index, ~] = predict(Mdl, x);
%view(Mdl.Trees{1}, 'Mode', 'graph')
scoreGrid = reshape(score(:, 1), size(x1, 1), size(x2, 2));
figure
h(1:2) = gscatter(x(:, 1), x(:, 2), y, 'rb');
hold on
%plot(x(svInd, 1), x(svInd, 2), 'ro', 'MarkerSize', 10)
contour(x1, x2, scoreGrid);
colorbar;
title('{\bf Random Forest Classification}')
xlabel('Mean width of Single points')
ylabel('Mean width of First points')
legend('+1: Laubbaum', '-1: Nadelbaum')
hold off

Index = str2double(Index);
ConfMat = confusionmat(y, Index)

TP = ConfMat(1, 1)
FP = ConfMat(1, 2)
FN = ConfMat(2, 1)
TN = ConfMat(2, 2)

Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = (2*Precision*Recall)/(Precision+Recall)
Accuracy = (TP+TN)/(TP+FP+FN+TN)
% Feature importance
imp = Mdl.OOBPermutedPredictorDeltaError;
figure;
bar(imp);
title('Feature importance Test');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
hold off
%viaLogisticRegression(beta, x(:, 1), x(:, 2), y, pct)
%mpgQuartiles = quantilePredict(Mdl, Index, 'Quantile', [0.25, 0.5, 0.75]);
figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

% Crossvalidation

indices = crossvalind('Kfold', y, 5) %all obj get individual numbers
AccuracyArray = []

for i=1:5
    test = (indices ==i);
    train = ~test ;%inversion of test

    TrainData = x(train, :);
    TrainLable = y(train, :);

    testData = x(test, :);
    testLable = y(test, :);

    Mdl = TreeBagger(40, TrainData, TrainLable);
    [Index, score] = predict(Mdl, testData);
    Index = str2double(Index);
    ConfMat = confusionmat(testLable, Index);

    % based on confussionsmat we can calculate FP, TN,  Overall
    TP = ConfMat(1, 1);
    FP = ConfMat(1, 2);
    FN = ConfMat(2, 1);
    TN = ConfMat(2, 2);
    Precision = TP/(TP+FP);
    Recall = TP/(TP+FN);
    Accuracy = (TP+TN)/(TP+FP+FN+TN);
    F1_Score = (2*Precision*Recall)/(Precision+Recall);
    AccuracyArray = [AccuracyArray,  Accuracy];
    %[cm,  order] = confusionmat(real_values, predicted_values)

end
MeanAccuracy = mean(AccuracyArray)*100
