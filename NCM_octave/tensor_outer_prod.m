function T = tensor_outer_prod(X)
m = size(X,1);
T = zeros(m,m,m);
for i = 1:m
   T(:,:,i) = X(:,1)*X(:,2)'*X(i,3);
end
end