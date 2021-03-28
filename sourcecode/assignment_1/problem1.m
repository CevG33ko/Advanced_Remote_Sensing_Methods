
%% Init Values
% Spruce Beech Fir Dead_tree
categories = ["Spruce", "Beech", "Fir", "Dead Tree"];
accuracy_table = [ 70 1 3 6;
                   3 64 12 8;
                   1 9 19 8;
                   5 3 4 34];
               
sum_classification = sum(accuracy_table,2); % the rows
sum_test_areas = sum(accuracy_table); % the columns

%% Calculate basic values
TP = diag(accuracy_table);
TN = calc_TN(accuracy_table);
FP = sum_classification - diag(accuracy_table);
FN = sum_test_areas' - diag(accuracy_table);

FP_rate = FP./sum_classification;
TP_rate = TP./(FP+FN);

sum_pixels = sum(sum_classification);

failure_1 = sum_classification - TP;
failure_2 = sum_test_areas - TP';

overall_accuracy = sum(TP)/sum_pixels * 100;

temp_cross_product_sum = sum(sum_classification.*sum_test_areas');

kappa = (sum_pixels * sum(TP) - temp_cross_product_sum)/(sum_pixels^2 - temp_cross_product_sum);

recall = TP./(TP+FN);    % completeness
precision = TP./(TP+FP); % correctness

F1_score = calc_f1(accuracy_table);

commision = 1 - TP./(TP+FP);
ommision = 1 - TP./(TP+FN);

disp(['Overall accuracy: ' , num2str(overall_accuracy)]);
disp(['Kappa: ' , num2str(kappa*100)]);

for i = 1:size(recall,1)
    disp(categories(i))
    disp([' recall: ' , num2str(recall(i)*100), ' Precision: ', num2str(precision(i) * 100)]);
    disp([' F1-Score: ' , num2str(F1_score(i)*100)]);
    disp([' Ommision: ' , num2str(ommision(i)*100), ' Commision: ', num2str(commision(i) * 100)]);
end
    disp('')
    disp(['B: ', num2str(accuracy_table(1,2))])
    disp('')
    disp(['C: ', num2str(FN(3))])
    disp('')
    disp(['D: ', num2str(FN(4))])
    disp('')
    disp(['E: ', num2str(FP(4))])
    disp('')
    disp(['D: (for firs)'])
    disp(['False positive rate: ', num2str(FP_rate(3)*100), ' True positive rate: ', num2str(TP_rate(3)*100)]);

