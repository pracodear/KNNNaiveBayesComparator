function [error_rate, false_pos_ratio] = ...
    calcError(X, Y, ws, wh, spam_prop, thresh)
%params: 
% X - features
% Y - targets
% ws - word freq in spam
% wh - word freq in ham
% spam_prop - assumed spam proportion
% thresh - classification threshold

false_neg = 0;
false_pos = 0;
numRows = size(X,1);
for i=1:numRows
    %instead of using long product of small numbers
    % use sums in the log domain, to avoid floating point underflow 
    ratio=log(spam_prop/(1-spam_prop)) + sum(log(ws(X(i,:))./wh(X(i,:))));        
    classif  = (ratio > thresh)*2-1;
    false_neg = false_neg + (classif==-1 && Y(i)==+1);
    false_pos = false_pos + (classif==+1 && Y(i)==-1);
end
error_rate = (false_neg + false_pos) / numRows;
num_of_hams = sum(Y==-1);
false_pos_ratio = false_pos / num_of_hams;
