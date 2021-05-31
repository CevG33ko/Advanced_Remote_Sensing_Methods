
N = 72;
s = 7; % from table
epsilon = 0.3; % relative number of outlier
p = 0.97; % propability of outlier free consensus set


%% Number of iterations

n =  log10(1 - p)/log10(1 - (1-epsilon)^s);

%% calculate the size of consensus set

S = (1-epsilon) * N; %N = anzahl aller Messungen

%% print

disp(['Number of iterations: ', num2str(n), ' -> ', num2str(ceil(n))])
disp(['Size of consensus set: ', num2str(S), ' -> ', num2str(ceil(S))])
