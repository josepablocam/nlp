#Author: Jose Pablo Cambronero (jpc485@nyu.edu)
#NLP Spring 2015 Final project
#ID: N17381190

##We use OpenNLPs MaxEntModel to train and tag our maxent model

import re
import tempfile
import subprocess


class Maxentmodel:
    def __init__(self, labels):
        self.labels = labels
        self.model_path = None
        self.START = 'START'
        self.LONGWORDLEN = 6
        self.OPENNLPCLASSPATH="/usr/local/jet/jet-all.jar:."
    
    def feat_map(self, word, prev_tag, prev_tag2):
        features = {}
        features["bias"] = True
        features["word"] = word
        features["suffix-4"] = word[-4:]
        features["suffix-3"] = word[-3:]
        features["suffix-2"] = word[-2:]
        features["last-letter"] = word[-1]
        features["firstCaps"] = word[0].isupper()
        features["has-number"] = re.search(r'[0-9]', word) != None
        features["is-long-word"] = len(word) > self.LONGWORDLEN
        features["tag_i-1"] = prev_tag
        features["tag_i-2"] = prev_tag2
        return features
            
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
        
    def build_features(self, tagged_corpus):
        featureset = []
        for sent in tagged_corpus:
            tags = []
            for word, tag in sent:
                prev_tag2, prev_tag = self.prev_tags(tags)
                features = self.feat_map(word, prev_tag, prev_tag2)
                featureset.append(features)
                tags.append(tag)
        return featureset


    def train(self, tagged_corpus, model_path, iter = 100):
        features = self.build_features(tagged_corpus)
        tags = [tag for sent in tagged_corpus for word, tag in sent]
        feat_file = tempfile.NamedTemporaryFile(mode="r+w", delete=False)
        for i in xrange(len(features)):
            str_feats = " ".join([feat + "=" + unicode(val) for feat,val in features[i].iteritems()])
            feat_file.write(str_feats.encode("utf-8"))
            feat_file.write(" " + tags[i])
            feat_file.write("\n")
        #run java trainer
        print "Calling java MaxEntModelTrain to train classifier, storing to %s" % model_path
        feat_file.seek(0)
        failure = subprocess.call(['java', '-cp', self.OPENNLPCLASSPATH, 'MaxEntModelTrain', feat_file.name, model_path])
        if not failure:
            self.model_path = model_path
        
        
    def tag_corpus(self, corpus):
        #create features for corpus
        #run java model
        #read in results
        #return in usual python structure
        
    