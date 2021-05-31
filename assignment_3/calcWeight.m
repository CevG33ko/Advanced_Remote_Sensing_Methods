function weight = calcWeight(grey_value_diff)
%% calcWeight function
    sigma = 14.72;
    weight = exp(-grey_value_diff^2/sigma^2);
end
