clear
XYZ = load("Stammpunkte.txt")
X=XYZ;
XY=XYZ(:,1:2);

Link = linkage(XYZ(:,1:2))
cutoff = median([Link(end-4,3) Link(end-2,3)]);
Cluster = cluster(Link,'Maxclust',3);
XYZ=[XYZ Cluster];
Cluster1 = XYZ(XYZ(:,4) == 1,:);

Cluster2 = XYZ(XYZ(:,4) == 2,:);
Cluster3 = XYZ(XYZ(:,4) == 3,:);

Line1 = ransacfitline(Cluster1(:,1:3).',1).';
Line2 = ransacfitline(Cluster2(:,1:3).',1).';
Line3 = ransacfitline(Cluster3(:,1:3).',1).';

dendrogram(Link,'ColorThreshold',1.2)
title('Dendogram')
ylabel('Linkage Distance')
xlabel('Cluster')

scatter3(Cluster1(:,1),Cluster1(:,2),Cluster1(:,3),'k')
hold on
scatter3(Cluster2(:,1),Cluster2(:,2),Cluster2(:,3),'r')
scatter3(Cluster3(:,1),Cluster3(:,2),Cluster3(:,3),'b')

plot3(Line1(:,1),Line1(:,2),Line1(:,3),'k');
plot3(Line2(:,1),Line2(:,2),Line2(:,3),'r');
plot3(Line3(:,1),Line3(:,2),Line3(:,3),'b');

vector1 = Line1(2,:) - Line1(1,:);
vector2 = Line2(2,:) - Line2(1,:);
vector3 = Line3(2,:) - Line3(1,:);

% Inclination
alpha1 = rad2deg(atan2(vector1(1,3),vector1(1,2)))
beta1 = rad2deg(atan2(vector1(1,3),vector1(1,1)))

alpha2 = rad2deg(atan2(vector2(1,3),vector2(1,2)))
beta2 = rad2deg(atan2(vector2(1,3),vector2(1,1)))

alpha3 = rad2deg(atan2(vector3(1,3),vector3(1,2)))
beta3 = rad2deg(atan2(vector3(1,3),vector3(1,1)))

title('Stem Estimation')
ylabel('Y-Coordinate')
xlabel('X-Coordinate')
zlabel('Z-Coordinate')
