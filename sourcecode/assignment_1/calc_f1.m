function f1 = calc_f1(confusion_matrix)
%CALC_F1 Summary of this function goes here
%   Detailed explanation goes here

sum_classification = sum(confusion_matrix,2); % the rows
sum_test_areas = sum(confusion_matrix); % the columns
               
TP = diag(confusion_matrix);
TN = calc_TN(confusion_matrix);
FP = sum_classification - diag(confusion_matrix);
FN = sum_test_areas' - diag(confusion_matrix);

f1 = (2*TP)./(2*TP+FN+FP);

end

