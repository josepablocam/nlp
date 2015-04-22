#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np
import itertools

class Ngrammodel:
    """ ngram-based POS tagger super class, has useful methods shared across n-gram models """

    def __init__(self, labels, map_unk, unk_ct = 2): #default to 2
        self.emit = None #emission probabilities
        self.trans = None #transition probabilities
        self.freq_words = None #frequent words, no need to remap
        self.START = 'START' #special start symbol
        self.STOP = 'STOP' # special stop symbol
        self.map_unk = map_unk #function to map unknown words
        self.unk_ct = unk_ct #threshold for unknown words
        self.labels = labels #list of labels possible
        
    def train_emit(self, corpus):
        emit = defaultdict(int) #empty dictionary for emissions 
        tag_freq = defaultdict(int) #empty dict for tags emitting word
        word_freq = defaultdict(int) #empty dict for word frequencies  
        
        for sent in corpus:
            for word, tag in sent:
                emit[(tag, word)] += 1
                tag_freq[tag] += 1
                word_freq[word] += 1
         
        #common enough words 
        self.freq_words = set([w for w,c in word_freq.iteritems() if c >= self.unk_ct]) 
         
        #re aggregate emissions based on unknown word mappings
        mapped_emit = defaultdict(float) #new dictionary for emission probs
        for (tag, word), ct in emit.iteritems():
            if word in self.freq_words:
                 mapped_emit[(tag, word)] = ct / float(tag_freq[tag])
            else:
                mapped_word = self.map_unk(word) 
                mapped_emit[(tag, mapped_word)] += ct / float(tag_freq[tag])
 
        #assign to members
        self.emit = dict(mapped_emit)
        

    #nice one liner from  http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    def make_ngram(self, n, input):
        return zip(*[input[i:] for i in range(n)])
        
    def count_ngrams(self, n, corpus):
        trans = defaultdict(float) #create empty dictionary for trans
        trans_denom = defaultdict(float) #normalizer for transitions
        ##ngram transitions in data
        for sent in corpus:
            tags = [tag for word, tag in sent]
            ext_tags = ([ self.START ] * (n - 1)) + tags + [ self.STOP ] #extend with special symbols
            ngrams = self.make_ngram(n, ext_tags) #create ngrams
            for ngram in ngrams:
                trans[ngram] += 1 #count ngrams
                trans_denom[ngram[ :-1]] += 1 #count denominator: dropping last in ngram
        return trans, trans_denom
        
    def laplace_smooth(self, trans, trans_denom):        
        ##we perform plus 1 smoothing for the ngram transitions
        ext_labels = self.labels + [ self.START, self.STOP ]
        number_of_tags = len(ext_labels)
        ngram_len = len(trans.iterkeys().next())
        for ngram in itertools.product(*[ext_labels for i in range(ngram_len)]):
            trans[ngram] += 1 ## add 1 smoothing
            trans[ngram] /= float(trans_denom[ngram[:-1]] + number_of_tags) #normalize

        return dict(trans)
    
    def train_trans(self, n, corpus):
        ngrams_cts, ngrams_denom_cts = self.count_ngrams(n, corpus)
        self.trans = self.laplace_smooth(ngrams_cts, ngrams_denom_cts)
     
                
    def train(self, n, corpus):
        self.train_emit(corpus)
        self.train_trans(n, corpus)
        
    def reword_sent(self, sentence):
        return [self.map_unk(word) if not word in self.freq_words else word for word in sentence]
    
    def safe_log(self, x):
        return log(x) if x != 0 else float('nan')
        
    def tag_corpus(self, corpus):
        clen = len(corpus)
        denom = float(clen)
        results = []
        for i in xrange(clen):
            results.append(self.tag_sent(corpus[i]))
            if i % 1000 == 0:
                print "%f done" %((i + 1) / denom)
        return results
     
    def confusion_matrix_sent(self, golden, predicted):
        confusion = defaultdict(float)
        matched = 0.0
        slen = len(golden)
        for i in xrange(slen):
            golden_tag = golden[i][1]
            predict_tag = predicted[i][1]
            pair = (golden_tag, predict_tag)
            confusion[pair] += 1
            matched += (golden_tag == predict_tag)
        
        return dict(confusion)

    def confusion_matrix_corpus(self, golden, predicted):
        corp_confusion = defaultdict(float)

        for g_sent, p_sent in zip(golden, predicted):
            sent_confusion = self.confusion_matrix_sent(g_sent, p_sent)
            for pair in sent_confusion:
                corp_confusion[pair] += sent_confusion[pair]

        return dict(corp_confusion)

