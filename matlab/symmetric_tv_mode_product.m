function T_f = symmetric_tv_mode_product(T,x,modes)
    
    % tensor mode product of symmetric tensor with vector
    % Input: T - tensor
    %        x - vector
    %        modes - number of modes for multiplication
    
    n_vec = size(T);
    if n_vec==1
        m = modes;
    else
        m = length(n_vec); % get order tensor
    end
    n = n_vec(1);      % get tensor dimensionality  
       
    T_mtx = reshape(T,n,n^(m-1));
    for i = 1:modes-1
       
       % multiply 
       T_mtx = x.'*T_mtx;
       
       % reshape
       T_mtx = reshape(T_mtx,n,n^(m-1-i));       
    end
    
    T_mtx = x.'*T_mtx;       
    
    if modes>m-2
        T_f = T_mtx.';
    else
        T_f = reshape(T_mtx,n*ones(1,m-modes));
    end
end
