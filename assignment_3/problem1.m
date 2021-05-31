%% Problem 1
%
%% Given Data
% RGB image

image = [ 22 21 50 22 23 51 30 31 33 ];

% degree_vector = [ 2 3 2 3 4 3 2 3 2 ];
% degree_matrix = diag(degree_vector);

graph = readmatrix('adjacent_matrix.txt');

weight_matrix = readmatrix('weight_matrix.txt');

degree_matrix = zeros(size(weight_matrix,2));

for i = 1:size(weight_matrix, 2)
    degree_matrix(i,i) = sum(weight_matrix(i,:));
end


%% calculate NCut values
A_1 = [1 2 4 5];
A_2 = [1 2 4 5];
A_3 = [3 6];

B_1 = [3 6];
B_2 = [7 8 9];
B_3 = [7 8 9];

ncut(A_1, B_1, weight_matrix)
ncut(A_2, B_2, weight_matrix)
ncut(A_3, B_3, weight_matrix)

%% Eigenvalues

DW = degree_matrix - weight_matrix;
[eVec, eVal] = eig(DW, degree_matrix);

y = eVec(:,2);

numerator = y' * DW * y;
denumerator = y' * degree_matrix * y ;

ncut_approx = numerator/denumerator;

ncut([1 2 4 5 7 8], [3, 6], weight_matrix)

%% Eigenvalues 1
weight_matrix1 = [weight_matrix(1,[1 2 3 4 5 6]);
                   weight_matrix(2,[1 2 3 4 5 6]);
                   weight_matrix(3,[1 2 3 4 5 6]);
                   weight_matrix(4,[1 2 3 4 5 6]);
                   weight_matrix(5,[1 2 3 4 5 6]);
                   weight_matrix(6,[1 2 3 4 5 6])]

degree_matrix1 = [degree_matrix(1,[1 2 3 4 5 6]);
                   degree_matrix(2,[1 2 3 4 5 6]);
                   degree_matrix(3,[1 2 3 4 5 6]);
                   degree_matrix(4,[1 2 3 4 5 6]);
                   degree_matrix(5,[1 2 3 4 5 6]);
                   degree_matrix(6,[1 2 3 4 5 6])]

DW1 = degree_matrix1 - weight_matrix1;
[eVec1, eVal1] = eig(DW1, degree_matrix1);

y1 = eVec1(:,2);
numerator1 = y1' * DW1 * y1;
denumerator1 = y1' * degree_matrix1 * y1 ;

ncut_approx1 = numerator1/denumerator1;

%% Eigenvalues 2
weight_matrix2 = [weight_matrix(1,[1 2 4 5 7 8 9]);
                   weight_matrix(2,[1 2 4 5 7 8 9]);
                   weight_matrix(4,[1 2 4 5 7 8 9]);
                   weight_matrix(5,[1 2 4 5 7 8 9]);
                   weight_matrix(7,[1 2 4 5 7 8 9]);
                   weight_matrix(8,[1 2 4 5 7 8 9]);
                   weight_matrix(9,[1 2 4 5 7 8 9])];

degree_matrix2 = [degree_matrix(1,[1 2 4 5 7 8 9]);
                   degree_matrix(2,[1 2 4 5 7 8 9]);
                   degree_matrix(4,[1 2 4 5 7 8 9]);
                   degree_matrix(5,[1 2 4 5 7 8 9]);
                   degree_matrix(7,[1 2 4 5 7 8 9]);
                   degree_matrix(8,[1 2 4 5 7 8 9]);
                   degree_matrix(9,[1 2 4 5 7 8 9])];
                   
DW2 = degree_matrix2 - weight_matrix2;
[eVec2, eVal2] = eig(DW2, degree_matrix2);

y2 = eVec2(:,2);
numerator2 = y2' * DW2 * y2;
denumerator2 = y2' * degree_matrix2 * y2 ;

ncut_approx2 = numerator2/denumerator2;

%% Eigenvalues 3
weight_matrix3 = [weight_matrix(3,[3 6 7 8 9]);
                   weight_matrix(6,[3 6 7 8 9]);
                   weight_matrix(7,[3 6 7 8 9]);
                   weight_matrix(8,[3 6 7 8 9]);
                   weight_matrix(9,[3 6 7 8 9])];

degree_matrix3 = [degree_matrix(3,[3 6 7 8 9]);
                   degree_matrix(6,[3 6 7 8 9]);
                   degree_matrix(7,[3 6 7 8 9]);
                   degree_matrix(8,[3 6 7 8 9]);
                   degree_matrix(9,[3 6 7 8 9])];

DW3 = degree_matrix3 - weight_matrix3;
[eVec3, eVal3] = eig(DW3, degree_matrix3);

y3 = eVec3(:,2);
numerator3 = y3' * DW3 * y3;
denumerator3 = y3' * degree_matrix3 * y3 ;

ncut_approx3 = numerator3/denumerator3;

%%

ncut_approx
y
ncut_approx1
y1
ncut_approx2
y2
ncut_approx3
y3

