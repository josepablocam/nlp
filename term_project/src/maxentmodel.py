#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190

##We use our model based on nltk's maximum entropy implementation
##to tag german POS
import nltk


class Maxentmodel:
    
    
    def __init__(self, labels):
        self.labels = labels
        self.model = None
        self.START = 'START'
        self.LONGWORDLEN = 6
    
    def feat_map(self, word, tag, prev_tag, prev_tag2):
        features = {}
        features["word"] = word
        features["suffix-4"] = word[-4:]
        features["suffix-3"] = word[-3:]
        features["last-letter"] = word[-1]
        features["firstCaps"] = word[0].isupper()
        features["is-long-word"] = len(word) > self.LONGWORDLEN
        features["tag_i-1"] = prev_tag
        features["tag_i-2"] = prev_tag2
        return (features, tag)

    def train(self, tagged_corpus):
        featureset = []
        for sent in tagged_corpus:
            tags = []
            for word, tag in sent:
                prev_tag2, prev_tag = self.prev_tags(tags)
                features = self.feat_map(word, tag, prev_tag, prev_tag2)
                featureset.append(features)
                tags.append(tag)           
        self.model = nltk.classify.MaxentClassifier.train(featureset, labels = self.labels, max_iter = 1000)
    
    #returns tuple of i -2 and i - 1 tags
    def prev_tags(self, tags):
        tlen = len(tags)
        if tlen == 0:
            prev_tag = prev_tag2 = self.START
        elif tlen == 1:
            prev_tag2 = self.START
            prev_tag = tags[-1]
        else:
            prev_tag2, prev_tag = tags[-2:]
        return prev_tag2, prev_tag
         
    
    #greedy search, not viterbi
    def tag_sent(self, sent):
        predicted = []
        for word,tag in sent:
            prev_tag2, prev_tag = self.prev_tags(predicted)
            features = self.feat_map(word, tag, prev_tag, prev_tag2)
            predicted.append(self.model.classify(features))
        return zip(sent, predicted)
        
    def tag_corpus(self, corpus):
        return [tag_sent(sent) for sent in corpus]
    
    
        
    
    
    
    
                
        
        
    


