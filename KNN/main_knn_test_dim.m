% KNN main script

% number of nearest neighbors
k_values = [1, 2, 10, 20, 50, 100];

% number of features we will use, i.e. dimentionality of the feature vectors
dim_values = [20, 40, 70, 100];

% different fractions of training data out of the total data
%TRAIN_FRACS = .1:.1:.9;
trainFrac = .5;
% number of runs per fraction (each run is with differnt training data
% which is chosen randomly)
RUNS_PER_K_VALUE = 10;
NUM_K_VALUES = length(k_values);
NUM_DIM_VALUES = length(dim_values);
DIRNAME ='../Data/enron1';

% error ratio on test set
testErrorMat     = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);
% false positive ratio on test set
testFalsePosMat  = zeros(RUNS_PER_K_VALUE, NUM_K_VALUES);

meanTestErrorMat = zeros(NUM_DIM_VALUES, NUM_K_VALUES);
meanTestFalsePosMat = zeros(NUM_DIM_VALUES, NUM_K_VALUES);

for idk_dim_value = 1:NUM_DIM_VALUES
    dim = dim_values(idk_dim_value)
    
    for idx_k_value = 1:NUM_K_VALUES
        k = k_values(idx_k_value)
        for run=1:RUNS_PER_K_VALUE
            display(run);
            fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
            train = importdata(fname);
            fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
            test  = importdata(fname);

            % the vectors without the labels
            trainVectors = train(:,1:dim);
            % the lables
            trainLabels = train(:,end);
            % use lables 1,-1 instead of 1,0
            trainLabels = 2*trainLabels - 1;

             % the vectors without the labels
            testVectors = test(:,1:dim);
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

    meanTestErrorMat(idk_dim_value, :) = mean(testErrorMat, 1);
    meanTestFalsePosMat(idk_dim_value, :) = mean(testFalsePosMat, 1);

end

colors = 'brgc';
dim_values_temp = cellstr(int2str(dim_values'))';
dim_values_str = arrayfun(@(x)strcat('dim=', x), dim_values_temp);

% prepare error rate graph

h = figure;
hold on;

for idk_dim_value = 1:NUM_DIM_VALUES
    plot(k_values, meanTestErrorMat(idk_dim_value, :), strcat(colors(idk_dim_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('Error rate');
legend(dim_values_str);
txt = sprintf('Error Rate\nTraining set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('error_rate_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);

% prepare false positive ratio graph

h = figure;
hold on;

for idk_dim_value = 1:NUM_DIM_VALUES
    plot(k_values, meanTestFalsePosMat(idk_dim_value, :), strcat(colors(idk_dim_value),'-o'));
end

xlabel('K value (# of neighbors)');
ylabel('False pos');
legend(dim_values_str);
txt = sprintf('False Positive Ratio\nTraining set size is %g of all data\nAverage of %d runs per K value', trainFrac, RUNS_PER_K_VALUE);
title(txt)
fname = sprintf('falso_pos_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
