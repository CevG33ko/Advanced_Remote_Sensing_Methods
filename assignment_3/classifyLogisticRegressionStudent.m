function [ output_args ] = classifyLogisticRegressionStudent( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

 load fisheriris
 
 % we just take two features from index 51 to 150
 
 features = meas(51:150,1:2);
 
 % setting the labels
 labels(1:50) = 1; labels(51:100) = 0; labels = labels';
 
 % plot the data
 scatter(features(1:51,1), features(1:51,2), '+g');
 hold on
 scatter(features(51:100,1), features(51:100,2), '+r');
 
 
 
end

