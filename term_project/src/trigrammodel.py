#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np
from ngrammodel import Ngrammodel

class Trigrammodel(Ngrammodel):
    """ trigram-based POS tagger """

    def __init__(self, labels, map_unk, unk_ct = 2): #default to 2
        Ngrammodel.__init__(self, labels, map_unk, unk_ct)
      
    def train(self, corpus):
        Ngrammodel.train(self, 3, corpus)
    
    ##TODO: reduce our search space!!!
    def tag_sent(self, sentence):
        pi = defaultdict(lambda: float('nan')) 
        pi[(-1, self.START, self.START)] = log(1) # viterbi probability initialize
        bp = dict()
        sentence = Ngrammodel.reword_sent(self, sentence) #replace any unknown word with categories
        s_len = len(sentence)
   
        wlabels = [ self.START ]
        ulabels = [ self.START ]

        for i in xrange(s_len):
            word = sentence[i] #current word
            for v in self.labels:
                vlabels = []
                if (v, word) in self.emit: # reduce our search space...only tags that we have seen for word w
                    vlabels.append(v) 
                    emission_prob = log(self.emit[(v, word)])
                    for u in ulabels:
                        probs = [ pi[(i - 1, w, u)] + log(self.trans[(w, u, v)]) for w in wlabels ]
                        #maximizer
                        pi[(i, u, v)] = np.nanmax(probs) + emission_prob #only add emission prob here, it's the same everywhere else
                        arg_max = np.nanargmax(probs)
                        if not isnan(arg_max):
                            bp[(i, u, v)] = wlabels[arg_max]
            wlabels = ulabels
            ulabels = vlabels

        #add in stop probabilities
        for u in ulabels:
            for v in self.labels:
                pi[(s_len - 1, u, v)] += Ngrammodel.safe_log(self, self.trans.get((u, v, self.STOP), 0))

        #decode
        u, v = max({(u, v) : p for (s, u , v), p in pi.iteritems() if s == s_len - 1 and not isnan(p)}.iteritems(), key = lambda x: x[1])[0] #predict last 2 tags
        #trace pointer for remaining predictions
        predicted = [ v, u] # create array of predictions, reverse order
        for i in range(s_len - 1, 1, -1):
            w = bp[(i, u, v)]
            predicted.append(w)
            v, u = u, w #swap

        return zip(sentence, predicted[::-1]) #reverse predictions and zip with sentence, return tuples

