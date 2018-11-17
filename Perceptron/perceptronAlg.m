% the feature vectors (i.e. the rows of featureVectors) should be normalized
% (after adding additional coordinate with -1).
% the labels should be 1 and -1.
function [error_ratio, false_positives_ratio, w] = perceptronAlg(initial_w, featureVectors, labels)

[num_of_items, num_of_features] = size(featureVectors);

w = initial_w;

mistakes = 0;
false_positives = 0;

for i = 1:num_of_items
    x = featureVectors(i, :);
    label = labels(i);
        
    decision = sign( dot(w, x) );
    if decision == 0
        decision = 1;
    end
    
    if decision ~= label
        w = w + label*x;
        
        mistakes = mistakes + 1;
        if label == -1
            false_positives = false_positives + 1;
        end
    end
end

num_of_hams = sum(labels == -1);
error_ratio = mistakes / num_of_items;
false_positives_ratio = false_positives / num_of_hams;
