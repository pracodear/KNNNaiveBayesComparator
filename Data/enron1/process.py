#!/usr/bin/python

import os, random
from pdb import set_trace, pm
from pprint import pprint
from string import punctuation
from operator import itemgetter
from collections import Counter

def process_class(files):
    all_words = Counter()
    data=[]
    for fpath in files:
        words_gen = (word.strip(punctuation).lower()
                     for line in open(fpath)
                     for word in line.split())        
        words  = Counter()
        fwords = Counter()
        for word in words_gen:
            if len(word) > 1:
                words[word] += 1
        nwords = float(sum(words.values()))
        for k,v in words.items():
            v = v/nwords
            fwords[k] = v
            
        nchars = 0.0
        maxrun = 0
        nruns = 0
        sumruns = 0.0
        chars_found = ''
        for line in open(fpath):
            runlen = 0
            #notice that '\n' is last char of line
            #so caps-run ends nicely at eol
            for c in line:
                nchars+=1
                if c in FEATURE_CHARS:
                    words[c] += 1
                chars_found += c
                if c.isupper():
                    runlen += 1
                elif runlen > 0:
                    maxrun = max(runlen, maxrun)
                    sumruns += runlen
                    nruns += 1
                    runlen = 0
        for k in chars_found:
            fwords[k] = words[k] / nchars
                    
        fwords['capital_run_length_average'] = float(sumruns)/nruns
        fwords['capital_run_length_longest'] = float(maxrun)
        fwords['capital_run_length_total'] = float(sumruns)

        #count appearance of word in mail only once
        bool_words = dict.fromkeys(words.keys(),1)
        all_words.update(bool_words)
        data.append(fwords)
    nrows = float(len(data))
    for k,v in all_words.items():
        v = v / nrows
        all_words[k] = v
    #top_words = sorted(words.items(), key=itemgetter(1), reverse=True)[:N]
    #set_trace()
    return data, all_words

def process_set(ham_files, spam_files, train_frac, count):
    ham, ham_total = process_class(ham_files)
    spam, spam_total = process_class(spam_files)
    ranks = Counter()
    both_total = spam_total + ham_total
    for k,freq in both_total.items():
        ham_freq = ham_total[k]
        spam_freq = spam_total[k]
        spamicity = spam_freq / (spam_freq+ham_freq)
        diff = abs(spam_freq - ham_freq)        
        if abs(spamicity -.5)>SPAMICITY_RADIUS and freq>RARE_THRESH:
            ranks[k] = diff
    #take N best in descending order
    selected = ranks.most_common(NFEATURES) if FEAT_SEL else both_total.items()
    f=open('stats_{0}_{1}.txt'.format(train_frac,count),'wt')
    for k,rank in selected:
        ham_freq = ham_total[k]
        spam_freq = spam_total[k]
        spamicity = spam_freq / (spam_freq+ham_freq)
        f.write('{0}, {1:g}, {2:g}, {3:g}, {4:g} \n'.format(
                k,ham_freq, spam_freq, rank, spamicity))
    f.close()
    #or: zip(*best)[0]
    features = [k for k,v in selected]
    if USE_CAP_RUN_STAT:
        features.extend(['capital_run_length_average',
                         'capital_run_length_longest',
                         'capital_run_length_total'])
    #[0] is ham class id, [1] is for spam
    ham_out  = prepare_output(ham,  features, '0')
    spam_out = prepare_output(spam, features, '1')
    return ham_out, spam_out, features
    
def prepare_output(rows, features, const_features):
    out = []
    for row in rows:
        out_row = ''
        out_row = ['{0:g},'.format(row[k]) for k in features]
        out_row += const_features
        out_row = ''.join(out_row)
        out.append(out_row + '\n')
    return out

def split_train_test(dirname, train_frac):
    paths = []
    for root, dirs, files in os.walk(dirname):
        paths.extend([os.path.join(root, name) for name in files])
    random.shuffle(paths)
    #paths = [os.path.join(dirname, fname) for fname in files]
    ntrain = int(round(train_frac * len(paths)))
    return paths[:ntrain], paths[ntrain:]

def one_run(train_frac, count):
    ham_train_paths, ham_test_paths = split_train_test(HAM, train_frac)
    spam_train_paths, spam_test_paths = split_train_test(SPAM, train_frac)

    train_ham, train_spam, features = process_set(
        ham_train_paths, spam_train_paths, train_frac, count)
    ham, ham_total = process_class(ham_test_paths)
    spam, spam_total = process_class(spam_test_paths)
    test_ham  = prepare_output(ham,  features, '0')
    test_spam = prepare_output(spam, features, '1')

    train = train_ham + train_spam
    test = test_ham + test_spam
    random.shuffle(train)
    random.shuffle(test)
    for x in 'train test'.split():
        f = open('{0}_{1}_{2}.txt'.format(x,train_frac,count),'wt')
        for row in eval(x):
            f.write(row)
        f.close()
        
#####################################################################
        
#folder names
HAM = 'ham'
SPAM = 'spam'

#careful: false means take all words(!)
FEAT_SEL = True
NFEATURES = 100

#ignore words with total freq lower than thresh
RARE_THRESH = 0.01

#max distance of spamicity from .5
SPAMICITY_RADIUS = 0.05
SPAMICITY_WEIGHT = 0.5

#add features of statistics about capital run-lengths
USE_CAP_RUN_STAT = False

#only count these chars as words (features)
FEATURE_CHARS=';([!$#'

train_fracs = [x/10.0 for x in range(1,10)]
for train_frac in train_fracs:
    for irun in range(10):
        one_run(train_frac, irun)
        print('  done run %d' % irun)
    print('done train_frac %g' % train_frac)
