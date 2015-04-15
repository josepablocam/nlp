#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190


from math import log, isnan
from collections import defaultdict
import numpy as np

class Bigrammodel:
    """ bigram-based POS tagger """

    
    def __init__(self, map_unk, unk_ct = 2): #default to 2
        self.emit = None #emission probabilities
        self.trans = None #transition probabilities
        self.word_freq = None #frequencies of words
        self.tag_freq = None # frequencies of tags
        self.START = 'START' #special start symbol
        self.STOP = 'STOP' # special stop symbol
        self.map_unk = map_unk #function to map unknown words
        self.unk_ct = unk_ct #threshold for unknown words
        self.labels = None #list of labels possible
        
    def train_emit(self, corpus):
        emit = defaultdict(int) #create empty dictionary for emissions 
        tag_freq = defaultdict(int) #create
        word_freq = defaultdict(int)  
        
        for sent in corpus:
            for word, tag in sent:
                emit[(tag, word)] = emit[(tag, word)] + 1
                tag_freq[tag] = tag_freq[tag] + 1
                word_freq[word] = word_freq[word] + 1

        #add cts for self.START/self.STOP tags
        tag_freq[self.START] = tag_freq[self.STOP] = len(corpus)
         
        #re aggregate emissions based on unknown word mappings
        mapped_emit = defaultdict(float)
        map_fun = self.map_unk
        threshold = self.unk_ct
        for (tag, word), ct in emit.iteritems():
            if word_freq[word] < threshold:
                 mapped_word = map_fun(word) 
                 mapped_emit[(tag, mapped_word)] = mapped_emit[(tag, mapped_word)] + (ct / float(tag_freq[tag]))
            else:
                 mapped_emit[(tag, word)] = ct / float(tag_freq[tag])
 
        #assign to members
        self.emit = dict(mapped_emit)
        self.word_freq = dict(word_freq)
        self.tag_freq = dict(tag_freq)
        #calculate labels
        self.labels = tag_freq.keys()
        
        
        
    def train_trans(self, corpus):
        trans = defaultdict(float) #create empty dictionary for trans
        ##bigram transitions
        for sent in corpus:
            tags = [tag for word, tag in sent]
            ext_tags = [ self.START ] + tags + [ self.STOP ] #extend with special symbols
            bigrams = zip(ext_tags, ext_tags[1:]) #create bigrams
            for bigram in bigrams:
                trans[bigram] += 1 #count bigrams
                
        ##we perform plus 1 smoothing for the bigram transitions
        number_of_tags = len(self.labels)
        
        for prev_tag in self.labels:
            for tag in self.labels:
                trans[(prev_tag, tag)] += 1   ##add 1 and then normalize
                trans[(prev_tag, tag)] /= float(self.tag_freq[prev_tag] + number_of_tags)
                
        #to prob and assign
        self.trans = dict(trans)
                
    def train(self, corpus):
        self.train_emit(corpus)
        self.train_trans(corpus)
        
    def reword_sent(self, sentence):
        return [self.map_unk(word) if self.word_freq.get(word, 0) < self.unk_ct else word for word in sentence]
    
    def _safe_log(self, x):
        return log(x) if x != 0 else float('nan')
    
    def tag_sent(self, sentence):
        pi = defaultdict(lambda: float('nan')) 
        pi[(-1, self.START)] = log(1) # viterbi probability initialize
        bp = dict()
        sentence = self.reword_sent(sentence) #replace any unknown word with categories
        s_len = len(sentence)
   
        for i in xrange(s_len):
            labels = self.labels + [ self.START ] if i == 0  else self.labels
            w = sentence[i] #current word
            for v in labels:
                emission_prob = self._safe_log(self.emit.get((v, w),0))
                if not isnan(emission_prob): #only try if possible, reduces our search space
                    probs = [ pi[(i - 1, u)] + emission_prob + self._safe_log(self.trans.get((u, v), 0)) for u in labels ]
                    #maximizer
                    pi[(i, v)] = np.nanmax(probs)
                    bp[(i, v)] = labels[np.nanargmax(probs)]

        #add in stop probabilities
        for v in labels:
            pi[(s_len - 1, v)] += self._safe_log(self.trans.get((v, self.STOP), 0))

        #decode
        last_tag = max({(s, v) : p for (s, v), p in pi.iteritems() if s == s_len - 1 and not isnan(p)}.iteritems(), key = lambda x: x[1])[0][1] #predict last tag
        predicted = [ last_tag ] # create array of predictions
        #trace pointer for remaining predictions
        for i in range(s_len - 1, 0, -1):
            predicted.append(bp[(i, predicted[-1])])

        return zip(sentence, predicted[::-1]) #reverse predictions and zip with sentence, return tuples
    
    def tag_sent_faster(self, sentence):
        pi = defaultdict(lambda: float('nan')) 
        pi[(-1, self.START)] = log(1) # viterbi probability initialize
        bp = dict()
        sentence = self.reword_sent(sentence) #replace any unknown word with categories
        s_len = len(sentence)
   
        for i in xrange(s_len):
            labels = self.labels + [ self.START ] if i == 0  else self.labels
            w = sentence[i] #current word
            for v in labels:
                if (v, w) in self.emit: # reduce our search space...only tags that we have seen for word w
                    emission_prob = log(self.emit[(v, w)])
                    probs = [ pi[(i - 1, u)] + log(self.trans[(u, v)]) for u in labels ]
                    #maximizer
                    pi[(i, v)] = np.nanmax(probs) + emission_prob #only add emission prob here, it's the same everywhere else
                    bp[(i, v)] = labels[np.nanargmax(probs)]

        #add in stop probabilities
        for v in labels:
            pi[(s_len - 1, v)] += self._safe_log(self.trans.get((v, self.STOP), 0))

        #decode
        last_tag = max({(s, v) : p for (s, v), p in pi.iteritems() if s == s_len - 1 and not isnan(p)}.iteritems(), key = lambda x: x[1])[0][1] #predict last tag
        predicted = [ last_tag ] # create array of predictions
        #trace pointer for remaining predictions
        for i in range(s_len - 1, 0, -1):
            predicted.append(bp[(i, predicted[-1])])

        return zip(sentence, predicted[::-1]) #reverse predictions and zip with sentence, return tuples
    
    
    def tag_corpus(self, corpus):
        return [self.tag_sent(sent) for sent in corpus]
     

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

