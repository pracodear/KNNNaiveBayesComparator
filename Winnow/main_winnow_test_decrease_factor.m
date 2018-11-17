% winnow main script

addpath('..');

% thresh is used in the classification, where we check
% the sign of: dot(w, x) - thresh
thresh = 20;

% increase and decrease factors
increase_factor = 2;
decrease_factor_values = [0.125, 0.25, 0.5];

% different fractions of training data out of the total data
%TRAIN_FRACS = .1:.1:.9;
trainFrac = .5;
% number of runs per fraction (each run is with differnt training data
% which is chosen randomly)
RUNS_PER_DECREASE_FACTOR = 10;
NUM_DECREASE_FACTOR_VALUES= length(decrease_factor_values);
DIRNAME ='../Data/enron1';

% error ratio on training set
trainErrorMat    = zeros(RUNS_PER_DECREASE_FACTOR, NUM_DECREASE_FACTOR_VALUES);
% error ratio on test set (when using constant weight vector - the one
% generated after training stage).
testErrorMat_const_w   = zeros(RUNS_PER_DECREASE_FACTOR, NUM_DECREASE_FACTOR_VALUES);
% error ratio on test set (when continuing updating the weight vector).
testErrorMat_updated_w = zeros(RUNS_PER_DECREASE_FACTOR, NUM_DECREASE_FACTOR_VALUES);
% false positive ratio on test set (when using constant weight vector - the
% one generated after training stage).
testFalsePosMat_const_w  = zeros(RUNS_PER_DECREASE_FACTOR, NUM_DECREASE_FACTOR_VALUES);
% false positive ratio on test set (when continuing updating the weight vector).
testFalsePosMat_updated_w  = zeros(RUNS_PER_DECREASE_FACTOR, NUM_DECREASE_FACTOR_VALUES);


for idx_decrease_factor_value = 1:NUM_DECREASE_FACTOR_VALUES
    decrease_factor = decrease_factor_values(idx_decrease_factor_value)
    for run=1:RUNS_PER_DECREASE_FACTOR
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
        
        % randomly mix the vectors in training set
        % -> no need anymore as the script process.py mixes them
        %perm = randperm(size(train, 1));
        %train = train(perm,:);
        
        % randomly mix the vectors in test set (as the algorithm continues
        % to learn)
        % -> no need anymore as the script process.py mixes them
        %perm = randperm(size(test, 1));
        %test = test(perm,:);        
        
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
       
        % initial weight vector
        num_of_features = size(trainVectors, 2);
        initial_w = ones(1, num_of_features);
        
        % perform the algorithm
        % w is the obtained weight vector
        [trainErrorMat(run,idx_decrease_factor_value) , ...
         false_positives_ratio, ...
         w ] ...
            = winnowAlg(initial_w, thresh, increase_factor, decrease_factor, trainVectors, trainLabels);
        
        % check on test set using the weight vector obtained from the
        % learning stage (don't update it anymore)
        [testErrorMat_const_w(run,idx_decrease_factor_value), ...
         testFalsePosMat_const_w(run,idx_decrease_factor_value) ] ...
            = hyperplaneClassify(w, thresh, testVectors, testLabels);
            
        % check on test set - continue updating the weight vector
        [testErrorMat_updated_w(run,idx_decrease_factor_value), ...
         testFalsePosMat_updated_w(run,idx_decrease_factor_value), ...
         w ] ...
            = winnowAlg(w, thresh, increase_factor, decrease_factor, testVectors, testLabels);
                
    end
end

meanTrainErrorMat = mean(trainErrorMat, 1);
meanTestErrorMat_const_w = mean(testErrorMat_const_w, 1);
meanTestFalsePosMat_const_w = mean(testFalsePosMat_const_w, 1);
meanTestErrorMat_updated_w = mean(testErrorMat_updated_w, 1);
meanTestFalsePosMat_updated_w = mean(testFalsePosMat_updated_w, 1);

h = figure; 
hold on;
plot(decrease_factor_values,meanTrainErrorMat, 'b-o');
plot(decrease_factor_values,meanTestErrorMat_const_w, 'r-o');
plot(decrease_factor_values,meanTestFalsePosMat_const_w, 'g-o');
plot(decrease_factor_values,meanTestErrorMat_updated_w, 'r-.o');
plot(decrease_factor_values,meanTestFalsePosMat_updated_w, 'g-.o');

display(decrease_factor_values);
display(meanTestFalsePosMat_updated_w);

xlabel('Decrease Factor');
ylabel('Error rate');
legend('Train', 'Test (const w)', 'false pos (const w)', 'Test (updated w)', 'false pos (updated w)');
txt = sprintf('Training set size is %g of all data\nIncrease factor = %g\nAverage of %d runs per training size', trainFrac, increase_factor, RUNS_PER_DECREASE_FACTOR);
title(txt)
fname = sprintf('results_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
