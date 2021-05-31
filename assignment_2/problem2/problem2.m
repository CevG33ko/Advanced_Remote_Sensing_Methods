%%% Practical Assignment 2
clearvars
close all
clc
format long g

% List filenames ascii files
% Loop 
files = dir('relativhoehen');
mean_int = [];
mean_pwidth_single=  [];
mean_pwidth_first = [];
mean_middle_to_first = [];
for j = 3:39
%filename = 'relativhoehen/93_Scheuereck_111_umring_12_NADEL_Seg993.ascii';
    file = files(j).name;
    folder = 'relativhoehen/';
    filename = strcat(folder,file);
     
    delimiterIn = ' ';
    headerlinesIn = 1;

    DataStruct = importdata(filename,delimiterIn,headerlinesIn);
    Points     = DataStruct.data(:,:);

    %divide heights into 10 layers

    Z = Points(:,3);
    interval = max(Z) - min(Z);
    step = interval /10;
    Z_ = [];
    for i = 1:10
        Z_{i}=  [Points((Z>=min(Z)+step*(i-1) & Z<min(Z)+step*i),:)];
    end
    %- Mean instensity upper layer (last one)
    mean_int = [mean_int; sum(Z_{10}(:,4))/length(Z_{10}(:,4))];
    %- Mean pulse width of single- und first-points for entire tree crown area
    Ptype = Points(:,6);
    single_pnts = Points((Ptype == 0),:);
    mean_pwidth_single =  [mean_pwidth_single; (sum(single_pnts(:,5))/length(single_pnts))];
    first_pnts = Points((Ptype == 1),:);
    mean_pwidth_first = [mean_pwidth_first; (sum(first_pnts(:,5))/length(first_pnts))];

    middle_pnts = Points((Ptype == 2),:);

    NumOfFirstPoints = size(first_pnts, 1);
    NumOfMiddlePoints = size(middle_pnts, 1);

    MeanMiddleToFirst = (NumOfMiddlePoints/NumOfFirstPoints);
    mean_middle_to_first = [mean_middle_to_first; MeanMiddleToFirst];

end

%% Mean Shift
dataPts = [mean_int*10, mean_pwidth_first];

[C_S, idx_S, V] = MeanShiftCluster(dataPts', 0.25, 2);

figure(123456)
clf
hold on

MeanShift1 = plot(C_S(1,1),C_S(2,1),'r*');
MeanShift2 = plot(C_S(1,2),C_S(2,2),'b*');

P1 = plot(dataPts([1:25],1),dataPts([1:25],2),'ro');
P2 = plot(dataPts([26:37],1),dataPts([26:37],2),'bo');

P1_est = plot(dataPts([idx_S == 1],1),dataPts([idx_S == 1],2),'r.');
P2_est = plot(dataPts([idx_S == 2],1),dataPts([idx_S == 2],2),'b.');
P3_est = plot(dataPts([idx_S == 3],1),dataPts([idx_S == 3],2),'g.');

legend([P1(1) P2(1) MeanShift1(1) MeanShift2() P1_est P2_est P3_est],'Laub Real','Nadel Real','Center Cluster 1','Center Cluster 2', 'Laub Classified', 'Nadel Classified', 'Not Classified');

title('2D Feature space Mean Shift')

figure(2)
hold on

%% Kmeans

[idx_K,C_K] = kmeans(dataPts(:,1:2), 2);

P1_real = plot(dataPts([1:25],1),dataPts([1:25],2),'ro');
P2_real = plot(dataPts([26:37],1),dataPts([26:37],2),'bo');

P1_est = plot(dataPts([idx_K == 2],1),dataPts([idx_K == 2],2),'r.');
P2_est = plot(dataPts([idx_K == 1],1),dataPts([idx_K == 1],2),'b.');


Kmeans1 = plot(C_K(1,1),C_K(1,2),'b*');
KmeansMean2 = plot(C_K(2,1),C_K(2,2),'r*');
legend([P1_real(1) P2_real(1) Kmeans1(1) KmeansMean2() P2_est P1_est],'Laub Real','Nadel Real','Center Cluster 1','Center Cluster 2', 'Laub Classified','Nadel Classified');

title('2D Feature space K-Mean')


%% Confusionmatrix
KnownGroup = zeros(1, 37);
KnownGroup(:,1:25) = 1;
KnownGroup(:,25:end) = 2;

Confus_S = confusionmat(KnownGroup,idx_S)
idx_K = idx_K.';
Confus_K = confusionmat(KnownGroup,idx_K)

pause

%% Additional

dataPts = [dataPts, mean_middle_to_first];

figure(3)
hold on


[C_S, idx_S, V] = MeanShiftCluster(dataPts', 0.5, 3);

P1_real = scatter3(dataPts([1:25],1), dataPts([1:25],2), dataPts([1:25],3),'r.');
P2_real = scatter3(dataPts([26:37],1), dataPts([26:37],2), dataPts([26:37],3),'b.');

P1_est = scatter3(dataPts([idx_S == 1],1), dataPts([idx_S == 1],2), dataPts([idx_S == 1],3), 'ro');
P2_est = scatter3(dataPts([idx_S == 2],1), dataPts([idx_S == 2],2), dataPts([idx_S == 2],3), 'bo');

KmeansMean1 = scatter3(C_S(1,1), C_S(2,1), C_S(3,1), 'r*');
KmeansMean2 = scatter3(C_S(1,2), C_S(2,2), C_S(3,2), 'b*');

legend([P1_real(1) P2_real(1) KmeansMean1(1) KmeansMean2() P2_est P1_est],'Laub Real','Nadel Real','Center Cluster 1','Center Cluster 2', 'Laub Classified','Nadel Classified');
title('3D Feature space Mean Shift');
xlabel('Intensity');
ylabel('Pulse width');
zlabel('Mean Number of reflections between first-points and last-points');

figure(4);
hold on

[idx_K,C_K] = kmeans(dataPts, 2);

P1_real = scatter3(dataPts([1:25],1), dataPts([1:25],2), dataPts([1:25],3),'b.');
P2_real = scatter3(dataPts([26:37],1), dataPts([26:37],2), dataPts([26:37],3),'r.');

P1_est = scatter3(dataPts([idx_K == 1],1), dataPts([idx_K == 1],2), dataPts([idx_K == 1],3), 'ro');
P2_est = scatter3(dataPts([idx_K == 2],1), dataPts([idx_K == 2],2), dataPts([idx_K == 2],3), 'bo');

C_K = C_K';

Kmeans1 = scatter3(C_K(1,1), C_K(2,1), C_K(3,1), 'r*');
Kmeans2 = scatter3(C_K(1,2), C_K(2,2), C_K(3,2), 'b*');

title('3D Feature space K-Mean');
legend([P1_real(1) P2_real(1) Kmeans1(1) KmeansMean2() P2_est P1_est],'Laub Real','Nadel Real','Center Cluster 1','Center Cluster 2', 'Laub Classified','Nadel Classified');
xlabel('Intensity');
ylabel('Pulse width');
zlabel('Mean Number of reflections between first-points and last-points')


Confus_S = confusionmat(KnownGroup,idx_S)
idx_K = idx_K.';
Confus_K = confusionmat(KnownGroup,idx_K)
