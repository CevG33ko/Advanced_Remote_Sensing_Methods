function [TN] = calc_TN(confusion_matrix)

size_matrix = size(confusion_matrix,2);

TN = zeros(size_matrix,1);

for i = 1:size_matrix
    TN(i) = sum(diag(confusion_matrix)) - confusion_matrix(i,i);
end

end

