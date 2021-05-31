fid = importdata('93_Scheuereck_111_umring_12_A_Laub_Prakrikum2_1.ascii')
IntPulsMatrix = fid.data(:,2:3); %Take only Int and Puls;
IntPulsMatrix(:,1)=10*IntPulsMatrix(:,1); %Multi Int by 10;
Matrix1 = IntPulsMatrix;
dataPts1 = Matrix1.';

fid = importdata('93_Scheuereck_111_umring_12_A_Nadel_Prakrikum2_1.ascii')
IntPulsMatrix = fid.data(:,2:3); %Take only Int and Puls;
IntPulsMatrix(:,1)=10*IntPulsMatrix(:,1); %Multi Int by 10;
Matrix2 = IntPulsMatrix;

dataPts2 = Matrix2.';

Matrix = [Matrix1; Matrix2];

dataPts = Matrix(:,1:2).';

[C, idx_S, V] = MeanShiftCluster(dataPts,0.15,2);

P1 = plot(Matrix1(:,1),Matrix1(:,2),"r*");
hold on
MeanShift1 = plot(C(1,1),C(2,1),"rO");
MeanShift2 = plot(C(1,2),C(2,2),"rO");

P2 = plot(Matrix2(:,1),Matrix2(:,2),"b*");
legend([P1(1) P2(1) MeanShift1(1) MeanShift2() ],'Laub','Nadel','Cluster 1','Cluster 2');
title('2D Feature space (* = Clusterzentrum Mean Shift)')
ylabel('Puls Weight of First Points')
xlabel('Intensity')
hold off

[idx_K,C] = kmeans(Matrix(:,1:2),2);

pause
K_Mean1 = plot(C(1,1),C(1,2),"kO");
hold on
K_Mean2 = plot(C(2,1),C(2,2),"kO");
P1 = plot(Matrix1(:,1),Matrix1(:,2),"r*");
P2 = plot(Matrix2(:,1),Matrix2(:,2),"b*");
%legend(K_Mean())
legend([P1(1) P2(1) K_Mean1(1) K_Mean2(1) ],'Laub','Nadel','Cluster 1','Cluster 2');
hold off


KnownGroup = zeros(1, 37);
KnownGroup(:,1:25) = 1;
KnownGroup(:,25:end) = 2;
Confus_S = confusionmat(KnownGroup,idx_S);
idx_K = idx_K.';
Confus_K = confusionmat(KnownGroup,idx_K);
title('2D Feature space (* = Clusterzentrum K-Mean)')
ylabel('Puls Weight of First Points')
xlabel('Intensity')
