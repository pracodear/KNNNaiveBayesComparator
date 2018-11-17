% adds dummy column of -1

function result = addDummyColumn(M)

newColumn = ones(size(M,1), 1) * (-1);

result = [newColumn M];
 