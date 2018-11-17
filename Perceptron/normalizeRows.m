% normalizes the rows of a matrix

function result = normalizeRows(M)

squares = M.^2;
norms = sqrt( sum(squares, 2) );
norms_matrix = repmat(norms, 1, size(M, 2));

result = M./norms_matrix;
