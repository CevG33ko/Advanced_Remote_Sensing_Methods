function [singlePoints, firstPoints] =filter_data(input_data)

singlePointsLogic = input_data(:,3) == 0;
singlePoints = input_data(singlePointsLogic, :);

firstPointsLogic = input_data(:,3) == 1;
firstPoints = input_data(firstPointsLogic, :);
