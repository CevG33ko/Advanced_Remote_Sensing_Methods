function assoc_ret = assoc(first, weight_matrix)

assoc_ret = 0;

for i = first
    assoc_ret = assoc_ret + sum(weight_matrix(i,:));
end

end

