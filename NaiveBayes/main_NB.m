%UCI
% data=importdata('../Data/UCI/spambase.data');
% remove "capical-run-length" statistics features
% so use only words appearances
% data = [data(:,1:end-4), data(:,end)];

global IGNORE_RARE_WORDS  IGNORE_COMMON ;

IGNORE_COMMON       = 0.1;
IGNORE_RARE_WORDS   = 0.01;
CLASSIF_THRESH      = 0;

THRESHOLDS=0:2:20;
NUM_THRESHOLDS=length(THRESHOLDS);
RUNS_PER_FRAC = 10;
TRAIN_FRACS = .1:.1:.9;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
NUM_TRAIN_FRACS
DIRNAME ='../Data/enron1';
testFalsePosMat  = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
trainErrorMat    = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
testErrorMat     = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
testFalsePosMat2 = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
trainErrorMat2   = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
testErrorMat2    = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
testFalsePosMatT = zeros(RUNS_PER_FRAC, NUM_THRESHOLDS);
trainErrorMatT   = zeros(RUNS_PER_FRAC, NUM_THRESHOLDS);
testErrorMatT    = zeros(RUNS_PER_FRAC, NUM_THRESHOLDS);
for iTrainFrac = 1:NUM_TRAIN_FRACS
    trainFrac = TRAIN_FRACS(iTrainFrac)
    for run=1:RUNS_PER_FRAC
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
        [trainErrorMat(run,iTrainFrac) , ...
            testErrorMat(run,iTrainFrac), ...
            testFalsePosMat(run,iTrainFrac) ] ...
            = naiveBayes(train,test, true, CLASSIF_THRESH);
        err = testErrorMat(run,iTrainFrac)
        [trainErrorMat2(run,iTrainFrac) , ...
            testErrorMat2(run,iTrainFrac), ...
            testFalsePosMat2(run,iTrainFrac) ] ...
            = naiveBayes(train,test, false, CLASSIF_THRESH);
        if trainFrac==0.7
            for i = 1:NUM_THRESHOLDS
                thresh = THRESHOLDS(i)
                [trainErrorMatT(run,i) , ...
                    testErrorMatT(run,i), ...
                    testFalsePosMatT(run,i) ] ...
                    = naiveBayes(train,test, true, thresh);
                if testFalsePosMatT(run,i)==0
                    display('0 false-pos!');
                end
            end
        end
    end
end

meanTestErrorMat = mean(testErrorMat, 1);
meanTrainErrorMat = mean(trainErrorMat, 1);
meanTestFalsePosMat= mean(testFalsePosMat, 1);

meanTestErrorMat2 = mean(testErrorMat2, 1);
meanTrainErrorMat2 = mean(trainErrorMat2, 1);
meanTestFalsePosMat2= mean(testFalsePosMat2, 1);

meanTestErrorMatT = mean(testErrorMatT, 1);
meanTrainErrorMatT = mean(trainErrorMatT, 1);
meanTestFalsePosMatT= mean(testFalsePosMatT, 1);

h = figure; 
hold on;
plot(THRESHOLDS,meanTestErrorMatT, 'r-o');
plot(THRESHOLDS,meanTrainErrorMatT, 'b-o');
plot(THRESHOLDS,meanTestFalsePosMatT, 'g-o');
xlabel('Threshold');
ylabel('Error rate');
legend('Test', 'Train', 'false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'thresh.fig');

h = figure; 
hold on;
plot(TRAIN_FRACS,meanTestErrorMat, 'r-o');
plot(TRAIN_FRACS,meanTrainErrorMat, 'b-o');
plot(TRAIN_FRACS,meanTestFalsePosMat, 'g-o');
plot(TRAIN_FRACS,meanTestErrorMat2, 'r-.o');
plot(TRAIN_FRACS,meanTrainErrorMat2, 'b-.o');
plot(TRAIN_FRACS,meanTestFalsePosMat2, 'g-.o');
xlabel('Training Fraction');
ylabel('Error rate');
legend('Test', 'Train', 'false pos', 'Test 2', 'Train 2', 'false pos 2');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'train_frac.fig');
