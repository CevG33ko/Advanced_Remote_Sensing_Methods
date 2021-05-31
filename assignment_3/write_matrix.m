
image = [ 22 21 50 22 23 51 30 31 33 ];

% adjacent_matrix = zeros(9, 9);

degree_vector = [ 2 3 2 3 4 3 2 3 2 ];
degree_matrix = diag(degree_vector);
% adjacent_matrix = adjacent_matrix + degree_matrix;
%
graph = readmatrix('adjacent_matrix.txt')
% adjacent_matrix(2,1) = 1;
% adjacent_matrix(3,1) = 0;
% adjacent_matrix(4,1) = 1;
% adjacent_matrix(5,1) = 0;
% adjacent_matrix(6,1) = 0;
% adjacent_matrix(7,1) = 0;
% adjacent_matrix(8,1) = 0;
% adjacent_matrix(9,1) = 0;
% % row2
% adjacent_matrix(3,2) = 1;
% adjacent_matrix(4,2) = 0;
% adjacent_matrix(5,2) = 1;
% adjacent_matrix(6,2) = 0;
% adjacent_matrix(7,2) = 0;
% adjacent_matrix(8,2) = 0;
% adjacent_matrix(9,2) = 0;
% % row3
% adjacent_matrix(4,3) = 0;
% adjacent_matrix(5,3) = 0;
% adjacent_matrix(6,3) = 1;
% adjacent_matrix(7,3) = 0;
% adjacent_matrix(8,3) = 0;
% adjacent_matrix(9,3) = 0;
% % row4
% adjacent_matrix(5,4) = 1;
% adjacent_matrix(6,4) = 0;
% adjacent_matrix(7,4) = 1;
% adjacent_matrix(8,4) = 0;
% adjacent_matrix(9,4) = 0;
% % row5
% adjacent_matrix(6,5) = 1;
% adjacent_matrix(7,5) = 0;
% adjacent_matrix(8,5) = 1;
% adjacent_matrix(9,5) = 0;
% % row6
% adjacent_matrix(7,6) = 0;
% adjacent_matrix(8,6) = 0;
% adjacent_matrix(9,6) = 1;
% % row7
% adjacent_matrix(8,7) = 1;
% adjacent_matrix(9,7) = 0;
% % row8
% adjacent_matrix(9,8) = 1;

% adjacent_matrix = adjacent_matrix + adjacent_matrix';
% adjacent_matrix = adjacent_matrix + degree_matrix;

%% Calculate the weightmatrix

% weight_matrix = zeros(size(graph, 1));
%
% for i =  1:size(graph, 1)
%     for j = i:size(graph,1)
%         if graph(i,j) == 1
%             grey_value_diff = abs(image(i) - image(j));
%             weight_matrix(i,j) = calcWeight(grey_value_diff);
%         end
%     end
% end
%
% weight_matrix = weight_matrix + weight_matrix' + diag(ones(1,size(weight_matrix,2))); % complete the weight matrix by adding the transverse and 1 at the diag
%
% writematrix(weight_matrix, 'weight_matrix');
