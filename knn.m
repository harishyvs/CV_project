
function [D, N] = knn(X, Y, k)
% [distance, index] = knn(query, data, k);


% [D, N] = sort(bsxfun(@plus,(-2)*(Y'*X),dot(Y,Y,1)'));
% N = N(1:k,:);
% D = D(1:k,:);

% top(x,k) is a partial sort function which returns top k entries of x 
[D, N] = top(bsxfun(@plus,(-2)*(Y'*X),dot(Y,Y,1)'), k);
D = bsxfun(@plus,D,dot(X,X,1));
return;



