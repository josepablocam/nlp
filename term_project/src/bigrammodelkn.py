#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np
from bigrammodel import Bigrammodel

class BigrammodelKN(Bigrammodel):
    """ bigram-based POS tagger with kneser-ney smoothing"""

    def __init__(self, labels, map_unk, unk_ct = 2, delta = 0.75): #default to 2
        Bigrammodel.__init__(self, labels, map_unk, unk_ct)
        self.delta = delta
      
    def train(self, corpus):
        Bigrammodel.train_emit(corpus)
        self.train_trans(self, corpus)

    def train_trans(self, corpus):
        bigram_cts, bigram_denom_cts = Bigrammodel.count_ngrams(2, corpus)
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
            if not prevtag in lambdas or not tag in continuation_cts:
                trans[bigram] = 0.0
            else:
                first_term = max(bigram_cts.get(bigram, 0) - self.delta, 0) / bigram_denom_cts[bigram[:-1]] #discount the term by delta
                second_term = lambdas[prevtag] * (continuation_cts[tag] / denom_continuations)
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
        
    def count_prior(self, bigram_cts):
        
        
    def calc_lambdas(self, bigram_denom_cts, prior_cts):
        lambdas = {}
        for tuple_tag, ct in bigram_denom_cts.iteritems():
            #we need to do this because of the way ngrammodel works, we generalize keys to tuple, which is annoying for the case when it's an unigram
            tag = tuple_tag[0] 
            lambdas[tag] = (self.delta / ct) * prior_cts[tag]
        return lambdas


            
            
        
    

