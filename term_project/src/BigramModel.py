class BigramModel:
    emit = None #emission probabilities
    trans = None #transition probabilities
    map_unk = None #function to map unknown words
    unk_ct = 1 #threshold for unknown words
    START = 'START' #special start symbol
    STOP = 'STOP' # special stop symbol
    labels = None #list of labels possible
    
    def __init__(map_unk, unk_ct = 1):
        this.map_unk = map_unk
        this.unk_ct = unk_ct
    
    def collect_tags(corpus):
        tags = set()
        for sent in corpus:
            for word, tag in sent:
                tags.add(tag)
        
        this.labels = list(tags)
    
        
    def train(corpus):
        emit = {} #create empty dictionary for emissions
        trans = {}
        tag_freq = {}
        word_freq = {}
        
        for sent in corpus:
            for word,tag in sent:
                emit[(tag, word)] = emit.get((tag, word), 0) + 1
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
                word_freq[word] = word_freq.get(word, 0) + 1
         
         #re aggregate emissions based on unknown word mappings
         mapped_emit = {}
         for (tag, word), ct in emit.iteritems():
             denom = float(tag_freq[tag])
             if word_freq[word] < unk_ct:
                 mapped_word = this.map_unk(word)
                 mapped_emit[(tag, mapped_word)] = (mapped_emit.get((tag, mapped_word), 0) + ct) / denom
             else:
                 mapped_emit[word] = ct / denom
           
         this.emit = mapped_emit
         
         ##bigram transitions
         for sent in corpus:
             tags = [for key,word in sent]
             ext_tags = [ START ] + tags + [ STOP ] #extend with special symbols
             bigrams = zip(ext_tags, ext_tags[1:])
             for bigram in bigrams:
                 denom = float(tag_freq[bigram[0]])
                 trans[bigram] = (trans.get(bigram, 0) + 1) / denom
         
         this.trans = trans   
                        
    def tag_sent(sentence):
        #implement viterbi
        
    
    def tag_corpus(corpus):
        result = []
        for sent in corpus:
            tags = tag_sent(sent)
            result.append(zip(sent, tags))
        return result
            
    def confusion_matrix_sent(golden, predicted):
        confusion = {}
        matched = 0.0
        for i in range(len(golden)):
            golden_tag = golden[i][1]
            predict_tag = predicted[i][1]
            pair = (golden_tag, predict_tag)
            confusion[pair] = confusion.get(pair, 0) + 1
            matched += (golden_tag == predict_tag)
        
        return confusion
        
    def confusion_matrix_corpus(golden, predicted):
        corp_confusion = {}
    
        for i in range(len(golden)):
            sent_confusion = confusion_matrix_sent(golden, predicted)
            for pair in set(corp_confusion.keys() + sent_confusion.keys()):
                corp_confusion[pair] = corp_confusion.get(pair, 0) + sent_confusion.get(pair, 0)
                
        matched = 0
        n_words = 0
        for ((key))
        return corp_confusion
        

def simple_unknown(word):
    return 'UNKNOWN'

def suffix_unknown(word):
    import re
    keit_heit = re.compile("\w+(keit|heit)$") #usually nouns
    lich_isch = re.compile("\w+(lich|isch)$") # usualy adjectives
    #TODO: ADD MORE
    if keit_heit.match(word) != None:
        return "HEIT_KEIT"
    else if lich_isch.match(word) != None:
        return "LICH_ISCH"
    else
        return "UNKNOWN"
    
    
    