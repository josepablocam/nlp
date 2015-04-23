#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np
from bigrammodel import Bigrammodel
import itertools

class BigrammodelKN(Bigrammodel):
    """ bigram-based POS tagger with kneser-ney smoothing"""

    def __init__(self, labels, map_unk, unk_ct = 2, delta = 0.75): #default to 2
        Bigrammodel.__init__(self, labels, map_unk, unk_ct)
        self.delta = delta
      
    def train(self, corpus):
        Bigrammodel.train_emit(self, corpus)
        self.train_trans(corpus)

    def train_trans(self, corpus):
        bigram_cts, bigram_denom_cts = Bigrammodel.count_ngrams(self, 2, corpus)
        self.trans = self.kn_smooth(bigram_cts, bigram_denom_cts)
       
    def kn_smooth(self, bigram_cts, bigram_denom_cts):
        prior_cts, continuation_cts = self.count_continuations_and_priors(bigram_cts)
        denom_continuations = sum(continuation_cts.values())
        lambdas = self.calc_lambdas(bigram_denom_cts, prior_cts)
        ext_labels = self.labels + [ self.START, self.STOP ]
        trans = {}
        ngram_len = len(bigram_cts.iterkeys().next())
        for bigram in itertools.product(*[ext_labels for i in range(ngram_len)]):
            prevtag, tag = bigram
            if prevtag == self.STOP or tag == self.START:
                trans[bigram] = 0.0 # these are impossible by design
            else:
                first_term = max(bigram_cts[bigram] - self.delta, 0.0) / bigram_denom_cts[bigram[:-1]] #discount the term by delta
                second_term = lambdas[prevtag] * (continuation_cts[tag] / float(denom_continuations))
                trans[bigram] = first_term + second_term
        return trans
                
    def count_continuations_and_priors(self, bigram_cts):
        continuations = defaultdict(set)
        priors = defaultdict(set)
        for prevtag, tag in bigram_cts:
            #add to sets
            continuations[tag].add(prevtag) #how many prevtags does tag follow (tag = continue)
            priors[prevtag].add(tag) #how many tags appear after prevtag (prevtag = prior)
        continuations = {tag : len(contexts) for tag, contexts in continuations.iteritems() }
        priors = {prevtag : len(contexts) for prevtag, contexts in priors.iteritems() }
        return priors, continuations
        
            
    def calc_lambdas(self, bigram_denom_cts, prior_cts):
        lambdas = {}
        for tuple_tag, ct in bigram_denom_cts.iteritems():
            #we need to do this because of the way ngrammodel works, we generalize keys to tuple, which is annoying for the case when it's an unigram
            tag = tuple_tag[0] 
            lambdas[tag] = (self.delta / ct) * prior_cts[tag]
        return lambdas


            
            
        
    

