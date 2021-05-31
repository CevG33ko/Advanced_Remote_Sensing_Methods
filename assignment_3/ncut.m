function ret = ncut(first, second, weight_matrix)
    cut_f_s = cut(first, second, weight_matrix);
    assoc_f = assoc(first, weight_matrix);
    assoc_s = assoc(second, weight_matrix);
    ret = cut_f_s/assoc_f + cut_f_s/assoc_s;
end
