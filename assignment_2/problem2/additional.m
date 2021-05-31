fid = importdata("93_Scheuereck_111_umring_12_A_Laub_Prakrikum2_1.ascii");
IntPulsMatrix = fid.data(:,2:4); %Take only Int and Puls
IntPulsMatrix(:,1)=10*IntPulsMatrix(:,1); %Multi Int by 10
Matrix1 = IntPulsMatrix;

fid = importdata("93_Scheuereck_111_umring_12_A_Nadel_Prakrikum2_1.ascii");
IntPulsMatrix = fid.data(:,2:4); %Take only Int and Puls
IntPulsMatrix(:,1)=10*IntPulsMatrix(:,1) ;%Multi Int by 10
Matrix2 = IntPulsMatrix;

Matrix = [Matrix1;Matrix2];

[idx_K,C] = kmeans(Matrix,2)
scatter3(Matrix(:,1),Matrix(:,2),Matrix(:,3),'.')
hold on
scatter3(C(:,1),C(:,2),C(:,3),'+')


title('3D Feature space (* = Clusterzentrum K-Mean)')
xlabel('Intensity')
ylabel('Puls Weight of First Points')
zlabel('Mean number of reflections between first-points and last-points')

hold off


KnownGroup = zeros(1, 37);
KnownGroup(:,1:25) = 1;
KnownGroup(:,25:end) = 2
idx_K = idx_K.'


Confus_K = confusionmat(KnownGroup,idx_K)

dataPts = Matrix.'; 

[C, idx_S, V] = MeanShiftCluster(dataPts,0.22,3)
C = C.'
scatter3(Matrix(:,1),Matrix(:,2),Matrix(:,3),'.')
hold on
scatter3(C(:,1),C(:,2),C(:,3),'+')
title('3D Feature space (* = Clusterzentrum Mean Shift)')
xlabel('Intensity')
ylabel('Puls Weight of First Points')
zlabel('Mean number of reflections between first-points and last-points')

hold off

Confus_S = confusionmat(KnownGroup,idx_S)

