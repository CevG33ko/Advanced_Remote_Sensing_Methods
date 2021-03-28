clc
clear all
close all
format long g

%Problem 5

A = [4565789.69, 5478954.73, 982.04, 235.67829145000];
R1 = [0, 0, 0, 235.67829151667];
R2 = [0, 0, 0, 235.67829153334];
r = [0.1; 0.05; -0.9937];
c = 3000000000;

%Reduce time vectors
R1(4) = R1(4)-A(4);
R2(4) = R2(4)-A(4);

%Calculate relative coordinates
increment = (c*r.*0.5)';
R1(1:3) = increment.*R1(4);
R2(1:3) = increment.*R2(4);

%Calculate absolute coordinates
R1(1:3) = R1(1:3)+A(1:3)
R2(1:3) = R2(1:3)+A(1:3)