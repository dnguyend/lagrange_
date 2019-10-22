function first_nan = find_eig_cnt(all_eig)
    first_nan = find(isnan(all_eig.x))
    if size(first_nan,1) == 0
        return nan
    else
      return first_nan[0]
end      
