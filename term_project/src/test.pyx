from math import log, isnan
from collections import defaultdict
import numpy as np

class Bigrammodel:
    """ bigram-based POS tagger """
    
    def __init__(self, map_unk, unk_ct = 2): #default to 2
        self.emit = None #emission probabilities
        self.trans = None #transition probabilities
        self.word_freq = None #frequencies of words
        self.START = 'START' #special start symbol
        self.STOP = 'STOP' # special stop symbol
        self.map_unk = map_unk #function to map unknown words
        self.unk_ct = unk_ct #threshold for unknown words
        self.labels = None #list of labels possible
        from math import log
    
    def train(self, corpus):
        emit = defaultdict(int) #create empty dictionary for emissions
        trans = defaultdict(float) #create empty dictionary for trans
        tag_freq = defaultdict(int) #create
        word_freq = defaultdict(int)
        
        for sent in corpus:
            for word, tag in sent:
                emit[(tag, word)] = emit[(tag, word)] + 1
                tag_freq[tag] = tag_freq[tag]+ 1
                word_freq[word] = word_freq[word] + 1

        #add cts for START/STOP tags
        tag_freq[self.START] = tag_freq[self.STOP] = len(corpus)
         
         #re aggregate emissions based on unknown word mappings
        mapped_emit = defaultdict(float)
        for (tag, word), ct in emit.iteritems():
            if word_freq[word] < self.unk_ct:
                 mapped_word = self.map_unk(word) 
                 mapped_emit[(tag, mapped_word)] = mapped_emit[(tag, mapped_word)] + (ct / float(tag_freq[tag]))
            else:
                 mapped_emit[(tag, word)] = ct / float(tag_freq[tag])
 
        #assign to members
        self.emit = dict(mapped_emit)
        self.word_freq = dict(word_freq)
        
        #calculate labels
        self.labels = tag_freq.keys()

        ##bigram transitions
        for sent in corpus:
            tags = [tag for word, tag in sent]
            ext_tags = [ self.START ] + tags + [ self.STOP ] #extend with special symbols
            bigrams = zip(ext_tags, ext_tags[1:])
            for bigram in bigrams:
                trans[bigram] += 1
                
        ##we perform plus 1 smoothing for the bigram transitions
        number_of_tags = len(self.labels)
        
        for prev_tag in self.labels:
            for tag in self.labels:
                trans[(prev_tag, tag)] += 1   ##add 1
                trans[(prev_tag, tag)] /= float(tag_freq[prev_tag] + number_of_tags) #normalize
                
        #to prob and assign
        self.trans = dict(trans)
        
    def reword_sent(self, sentence):
        return [self.map_unk(word) if self.word_freq.get(word, 0) < self.unk_ct else word for word in sentence]
    
    def __safe_log__(self, x):
        return log(x) if x != 0 else float('nan')
    
    
    def tag_sent(self, sentence):
        pi = defaultdict(lambda: float('nan')) 
        pi[(-1, self.START)] = log(1) # viterbi probability initialize
        bp = dict()
        sentence = self.reword_sent(sentence) #replace any unknown word with categories
        s_len = len(sentence)
   
        for i in xrange(s_len):
            labels = self.labels + [ self.START] if i == 0  else self.labels
            w = sentence[i] #current word
            for v in labels:
                emission_prob = self.__safe_log__(self.emit.get((v, w),0))
                if not isnan(emission_prob): #only try if possible
                    probs = [ pi[(i - 1, u)] + emission_prob + self.__safe_log__(self.trans.get((u, v), 0)) for u in labels ]
                    #maximizer
                    pi[(i, v)] = np.nanmax(probs)
                    bp[(i, v)] = labels[np.nanargmax(probs)]

        #add in stop probabilities
        for v in labels:
            pi[(s_len - 1, v)] += self.__safe_log__(self.trans.get((v, self.STOP), 0))

        #decode
        last_tag = max({(s, v) : p for (s, v), p in pi.iteritems() if s == s_len - 1 and not isnan(p)}.iteritems(), key = lambda x: x[1])[0][1] #predict last tag
        predicted = [ last_tag ] # create array of predictions
        #trace pointer for remaining predictions
        for i in range(s_len - 1, 0, -1):
            predicted.append(bp[(i, predicted[-1])])

        return zip(sentence, predicted[::-1]) #reverse predictions and zip with sentence, return tuples
        
    
    def tag_corpus(self, corpus):
        return [self.tag_sent(sent) for sent in corpus]
        
        
def testrun():
    for i in xrange(len(test.doc_without_tags)):
        if(i % 1000 == 0):
            print "running %d" % i
        res = test.m.tag_sent(test.doc_without_tags[i])
        






from tigerutil import *
doc = tigertsv_to_list("/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv")

m = Bigrammodel(lambda x: "UNKNOWN", 2)
#m.train(doc)

#doc_without_tags = [[word for word,tag in sent] for sent in doc]
#m.tag_sent(doc_without_tags[10])


#
#     def confusion_matrix_sent(golden, predicted):
#         confusion = {}
#         matched = 0.0
#         for i in range(len(golden)):
#             golden_tag = golden[i][1]
#             predict_tag = predicted[i][1]
#             pair = (golden_tag, predict_tag)
#             confusion[pair] = confusion.get(pair, 0) + 1
#             matched += (golden_tag == predict_tag)
#
#         return confusion
#
#     def confusion_matrix_corpus(golden, predicted):
#         corp_confusion = {}
#
#         for i in range(len(golden)):
#             sent_confusion = confusion_matrix_sent(golden, predicted)
#             for pair in set(corp_confusion.keys() + sent_confusion.keys()):
#                 corp_confusion[pair] = corp_confusion.get(pair, 0) + sent_confusion.get(pair, 0)
#
#         matched = 0
#         n_words = 0
#         for ((key))
#         return corp_confusion
#
#
# def simple_unknown(word):
#     return 'UNKNOWN'
#
# def suffix_unknown(word):
#     import re
#     keit_heit = re.compile("\w+(keit|heit)$") #usually nouns
#     lich_isch = re.compile("\w+(lich|isch)$") # usualy adjectives
#     #TODO: ADD MORE
#     if keit_heit.match(word) != None:
#         return "HEIT_KEIT"
#     else if lich_isch.match(word) != None:
#         return "LICH_ISCH"
#     else
#         return "UNKNOWN"
#
#
#