function cut = cut(first, second, weight_matrix)

%% Calculates the weight between first and second
%
% IN: 
% first = A one dimensional array of nodes in the first segment
% second = A one dimensional array of nodes in the second segment
% weight_matrix = the weight matrix describing the whole graph

cut = 0;

for i = first
    for j = second
        cut = cut + weight_matrix(i,j);
    end
end

end
