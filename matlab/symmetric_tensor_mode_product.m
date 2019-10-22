function T = symmetric_tensor_mode_product(T,X,modes)
    
    n_vec = size(T);   
    m = length(n_vec); % get order tensor    
    n = n_vec(1);      % get tensor dimensionality  
    n_2 = size(X,2);   % get new tensor dimensionality    
    for i = 1:length(modes)
       mode = modes(i); 
       mode_vec = [mode,1:mode-1,mode+1:m];
       n_vec = n_vec(mode_vec);
       n_vec(1) = n_2;
       T = permute(T,mode_vec);       
       T_mtx = X'*reshape(T,n,[]);       
       mode_vec = [2:mode,1,mode+1:m];       
       T = permute(reshape(T_mtx,n_vec),mode_vec);
       n_vec = n_vec(mode_vec);
    end
end

