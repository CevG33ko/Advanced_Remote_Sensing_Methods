%% A

beeches = normcdf([-1 1 1.5], 0, 1);
beeches(2) - beeches(1);
beeches_tn = beeches(3);
beeches_fp = 1 - beeches_tn;

spruces = normcdf([-1 1 1.5], 2.5, 1);
spruces(2) - spruces(1);
spruces_fn = spruces(3);
spruces_tp = 1 - spruces_fn;

disp(['TP: ', num2str(spruces_tp)])
disp(['FP: ', num2str(beeches_fp)])
disp(['FN: ', num2str(spruces_fn)])
disp(['TN: ', num2str(beeches_tn)])

%% B

beeches = normcdf(-6:0.1:6, 0, 1);
spruces = normcdf(-6:0.1:6, 2.5, 1);

number_of_values = size(beeches, 1);
temp_ones = ones(1, number_of_values);

TP = temp_ones - spruces;
FP = temp_ones - beeches;

plot(FP, TP)
hold on
plot(0:0.1:1, 0:0.1:1)

title('ROC-Curve')
xlabel('False Positive Rate [%]') 
ylabel('True Positive Rate [%]') 
