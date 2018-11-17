% this function expects the labels to be 1 and -1.
function [error_ratio, false_positives_ratio, w] = winnowAlg(initial_w, thresh, increase_factor, decrease_factor, featureVectors, labels)

[num_of_items, num_of_features] = size(featureVectors);

features_avg = mean(featureVectors);

w = initial_w;

mistakes = 0;
false_positives = 0;

for i = 1:num_of_items
    x = featureVectors(i, :);
    label = labels(i);
        
    decision =  sign( dot(w, x) - thresh );
    if decision == 0
        decision = 1;
    end
    
    if decision ~= label
        active_features =  x > 10*features_avg;
        
        %if label was 1 we have to multiply by increase_factor
        %if label was -1 we have to multiply by decrease_factor
        if label == 1
        	factor = increase_factor;
        else
        	factor = decrease_factor;
        end
        
        w(active_features) = w(active_features) * factor;
        
        mistakes = mistakes + 1;
        if label == -1
            false_positives = false_positives + 1;
        end
    end
end

num_of_hams = sum(labels == -1);
error_ratio = mistakes / num_of_items;
false_positives_ratio = false_positives / num_of_hams;
