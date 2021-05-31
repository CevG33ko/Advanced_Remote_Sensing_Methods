Data = importdata('test_data_baumarten_group #5.mat');
x = Data.instance_matrix;
y = Data.label_vector;

y(y==-1) = 0;

[beta_,  dev,  status] = glmfit(x, y, 'binomial', 'logit') %fit the logistic regresion into two classes



Labels_prep = glmval(beta_, x, 'logit')>=0.5 ;%predict lables
Labels_prep = double(Labels_prep);
ConfMat = confusionmat(y, Labels_prep)
TP = ConfMat(1, 1)
FP = ConfMat(1, 2)
FN = ConfMat(2, 1)
TN = ConfMat(2, 2)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = (2*Precision*Recall)/(Precision+Recall)
Accuracy = (TP+TN)/(TP+FP+FN+TN)

pct = sum(y==Labels_prep)/length(y)*100;
visLogisticRegression(beta_, x(:, 1), x(:, 2), y, pct);

%% Cross validation
indices = crossvalind('Kfold', y, 5); %all obj get individual numbers
AccuracyArray = [];

for i=1:5
    test = (indices == i);
    train = ~test; %inversion of test
    TrainData = x(train, :);
    TrainLable = y(train, :);
    testData = x(test, :);
    testLable = y(test, :);
    [beta_, dev, status] = glmfit(TrainData, TrainLable, 'binomial', 'logit');
    %based on confussionsmat we can calculate FP, TN,  Overall
    Labels_prep = glmval(beta_, testData, 'logit')>=0.5; %predict lables
    Labels_prep = double(Labels_prep);
    ConfMat = confusionmat(testLable, Labels_prep);
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
    %
end

MeanAccuracy = mean(AccuracyArray)*100
