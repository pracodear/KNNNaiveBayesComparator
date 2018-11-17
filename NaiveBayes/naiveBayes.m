function [trainError, testError, testFalsePos] = ...
    naiveBayes(trainData, testData, not_biased, thresh)

global IGNORE_RARE_WORDS IGNORE_COMMON;

%target
Y = 2*int8(trainData(:,end))-1;
testY = 2*int8(testData(:,end))-1;
%input features
X = trainData(:,1:end-1);
testX = testData(:,1:end-1);
X = (X>0);
testX = (testX>0);

spam = X(Y==+1,:);
ham  = X(Y==-1,:);
ws = mean(spam);
wh = mean(ham);

%avoid zero in probability formula
%assume 250,000 English words, and ~100 words-message
%so general probability of a word not in message is ((n-1)*n)^100
HAM_DICT_SIZE = 250000
%in spam they also "twist" words
SPAM_DICT_SIZE = HAM_DICT_SIZE* 10;
AVG_WORDS_IN_MSG = 100;
HAM_WORD_PROB = 1 -...
    ((HAM_DICT_SIZE-1)/HAM_DICT_SIZE)^AVG_WORDS_IN_MSG;
SPAM_WORD_PROB = 1 -...
    ((SPAM_DICT_SIZE-1)/SPAM_DICT_SIZE)^AVG_WORDS_IN_MSG;
wh(wh==0) = HAM_WORD_PROB;
ws(ws==0) = SPAM_WORD_PROB;

%init feature-selector to true
ind = true(1,size(ws,2));
%ignore non-discriminative common words
if IGNORE_COMMON>0
    %common = (abs(min(ws,wh)./max(ws,wh)-1)>IGNORE_COMMON);
    common  = abs(ws./(ws+wh) - .5) > IGNORE_COMMON;
    ncommon = sum(~common);
    if ncommon>0
        sprintf('eliminated %d common (of %d features)', ncommon, sum(ind))
        ind = ind & common;
    end
end
if IGNORE_RARE_WORDS>0
    rare = (ws+wh>IGNORE_RARE_WORDS);
    nrare= sum(~rare);
    if nrare>0
        sprintf('eliminated %d rare (of %d features)', nrare, sum(ind))
        ind = ind & rare;
    end
end

ws = ws(ind);
wh = wh(ind);
X = X(:,ind);
testX = testX(:,ind);

trainSpamProp = mean(trainData(:,end));
testSpamProp  = mean(testData(:,end));
if not_biased
    testSpamProp  = .5;
    trainSpamProp = .5;
end
[trainError, dummy] = calcError(X,Y,ws,wh,trainSpamProp, thresh);
[testError, testFalsePos ] = calcError(testX,testY,ws,wh,testSpamProp,thresh);
