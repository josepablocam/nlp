#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np
from ngrammodel import Ngrammodel

class Bigrammodel(Ngrammodel):
    """ bigram-based POS tagger """

    def __init__(self, labels, map_unk, unk_ct = 2): #default to 2
        Ngrammodel.__init__(self, labels, map_unk, unk_ct)
      
    def train(self, corpus):
        Ngrammodel.train(self, 2, corpus)

    def tag_sent(self, sentence):
        pi = defaultdict(lambda: float('nan')) 
        pi[(-1, self.START)] = log(1) # viterbi probability initialize
        bp = dict()
        sentence = Ngrammodel.reword_sent(self, sentence) #replace any unknown word with categories
        s_len = len(sentence)
   
        ulabels = [ self.START ]
        
        for i in xrange(s_len):
            w = sentence[i] #current word
            vposs = [] #store possible v tags for word to use in next iteration
            for v in self.labels:
                if (v, w) in self.emit: # reduce our search space...only tags that we have seen for word w
                    vposs.append(v) #keep track of these
                    emission_prob = log(self.emit[(v, w)])
                    probs = [ pi[(i - 1, u)] + log(self.trans[(u, v)]) for u in ulabels ]
                    #maximizer
                    pi[(i, v)] = np.nanmax(probs) + emission_prob #only add emission prob here, it's the same everywhere else
                    bp[(i, v)] = ulabels[np.nanargmax(probs)]
            ulabels = vposs #only worth searching through u's that were possible

        #add in stop probabilities
        for v in self.labels:
            pi[(s_len - 1, v)] += Ngrammodel.safe_log(self, self.trans.get((v, self.STOP), 0))

        #decode
        last_tag = max({v : p for (s, v), p in pi.iteritems() if s == s_len - 1 and not isnan(p)}.iteritems(), key = lambda x: x[1])[0] #predict last tag
        predicted = [ last_tag ] # create array of predictions
        #trace pointer for remaining predictions
        for i in range(s_len - 1, 0, -1):
            predicted.append(bp[(i, predicted[-1])])

        return zip(sentence, predicted[::-1]) #reverse predictions and zip with sentence, return tuples

