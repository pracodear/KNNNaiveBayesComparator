% KNN main script

% number of nearest neighbors
k_values = [1, 2, 10, 20, 50, 100];

% different fractions of training data out of the total data
%TRAIN_FRACS = .1:.1:.9;
trainFrac = .5;
% number of runs per fraction (each run is with differnt training data
% which is chosen randomly)
RUNS_PER_K_VALUE = 10;
NUM_K_VALUES = length(k_values);
DIRNAME ='../Data/enron1';

% error ratio on test set
testErrorMat     = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);
% false positive ratio on test set
testFalsePosMat  = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);

for idx_k_value = 1:NUM_K_VALUES
    k = k_values(idx_k_value)
    for run=1:RUNS_PER_K_VALUE
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
                
        % the vectors without the labels
        trainVectors = train(:,1:end-1);
        % the lables
        trainLabels = train(:,end);
        % use lables 1,-1 instead of 1,0
        trainLabels = 2*trainLabels - 1;
        
         % the vectors without the labels
        testVectors = test(:,1:end-1);
        % the lables
        testLabels = test(:,end);
        % use lables 1,-1 instead of 1,0
        testLabels = 2*testLabels - 1;
        
                
        % classify the test set
        [testErrorMat(run,idx_k_value), ...
         testFalsePosMat(run,idx_k_value)] ...
            = knnClassify(trainVectors, trainLabels, testVectors, testLabels, k, 0.5);
                
    end
end

meanTestErrorMat = mean(testErrorMat, 1);
meanTestFalsePosMat= mean(testFalsePosMat, 1);

h = figure; 
hold on;
plot(k_values,meanTestErrorMat, 'r-o');
plot(k_values,meanTestFalsePosMat, 'g-o');
xlabel('K value (# of neighbors)');
ylabel('Error rate');
legend('Test', 'false pos');
txt = sprintf('Training set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('results_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
