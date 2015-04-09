class model:
    emit = None #emission probabilities
    trans = None #transition probabilities
    map_unk = None #function to map unknown words
    unk_ct = 1 #threshold for unknown words
    START = 'START' #special start symbol
    STOP = 'STOP' # special stop symbol
    labels = None #list of labels possible
    
    def __init__(self, map_unk, unk_ct = 1):
        self.map_unk = map_unk
        self.unk_ct = unk_ct
             
    def train(self, corpus):
        emit = {} #create empty dictionary for emissions
        trans = {}
        tag_freq = {}
        word_freq = {}
        
        for sent in corpus:
            for word, tag in sent:
                emit[(tag, word)] = emit.get((tag, word), 0) + 1
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
                word_freq[word] = word_freq.get(word, 0) + 1

        #add cts for START/STOP tags
        tag_freq[self.START] = tag_freq[self.STOP] = len(corpus)
         
         #re aggregate emissions based on unknown word mappings
        mapped_emit = {}
        for (tag, word), ct in emit.iteritems():
         if word_freq[word] < self.unk_ct:
             mapped_word = self.map_unk(word) 
             mapped_emit[(tag, mapped_word)] = mapped_emit.get((tag, mapped_word), 0) + ct
         else:
             mapped_emit[word] = ct
 
        #to prob and assign
        self.emit = { (tag, word): ct / float(tag_freq[tag])) for (tag, word), ct in emit.iteritems() }

        ##bigram transitions
        for sent in corpus:
         tags = [tag for word, tag in sent]
         ext_tags = [ self.START ] + tags + [ self.STOP ] #extend with special symbols
         bigrams = zip(ext_tags, ext_tags[1:])
         for bigram in bigrams:
             trans[bigram] = trans.get(bigram, 0) + 1

        #to prob and assign
        self.trans = { (tag1, tag2): ct / float(tag_freq[tag1])) for (tag1, tag2), ct in trans.iteritems() }


    def tag_sent(self, sentence):
        ##If we haven't gotten a list of tags, calculate it
        if self.labels == None:
            self.labels = list(set([tag for (tag, word) in self.emit.iteritems()]))
        
        pi = { (-1, self.START): 1.0 } #viterbi probability intialize
        bp = {} #backpointer
        s_len = len(sentence)
          
        for i in range(s_len):
            labels = self.labels + [ self.START ] if i == 1 else self.labels
            for v in labels:
                w = sentence[i] #current word
                #possible prior tags for (v → w)
                probs = [ (u, pi.get((i - 1, u), 0.0) * self.emit.get((v, w), 0.0) * self.trans.get((u, v), 0.0) for u in labels ]
                #maximizer
                argmax, max_prob = max(probs.iteritems(), key = lambda x: x[1])
                pi[(i, v)] = max_prob
                bp[(i, v)] = argmax
                
        #add in stop probabilities
        for v in labels:
            pi[(s_len - 1, v)] *= pi[(s_len - 1, v)] * self.trans[(v, self.STOP)]
        
        #decode
        predicted = max({(s, v) : p for (s, v), p in pi.iteritems() if s == s_len - 1}, key = lambda x: x[1])[0] #predict last tag
        #trace pointer for remaining predictions
        for i in range(s_len - 1, 0, -1):
            predicted.add(bp[(i, predicted[-1])])
        
        return zip(sentence, labels[::-1]) #reverse predictions and zip with sentence, return tuples
            


#
#
#     def tag_corpus(corpus):
#         result = []
#         for sent in corpus:
#             tags = tag_sent(sent)
#             result.append(zip(sent, tags))
#         return result
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