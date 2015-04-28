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
        if prev_tag != None and prev_tag2 != None:
            features["tag_i-1"] = prev_tag
            features["tag_i-2"] = prev_tag2
        return features
            
    def build_features_sent(self, sent):
        featureset = []
        tags = [self.START, self.START]
        prev_tag2 = prev_tag = None
        for word, tag in sent:
            if tag != None:
                prev_tag2, prev_tag = tags[-2:]
            features = self.feat_map(word, prev_tag, prev_tag2)
            featureset.append(features)
            tags.append(tag)
        return featureset


    def features_str(self, features):
        str = " ".join([feat + "=" + unicode(val) for feat,val in features.iteritems()])
        return str.encode("utf-8")


    def train(self, tagged_corpus, model_path, iter = 100):
        feature_matrix = []
        realized_tags = []
        for sent in tagged_corpus:
            for word,tag in sent:
                realized_tags.append(tag) #collect realized tags
            sent_features = self.build_features_sent(sent)
            for word_features in sent_features:
                feature_matrix.append(word_features)
        feat_file = tempfile.NamedTemporaryFile(mode="r+w")
        for i in xrange(len(feature_matrix)):
            str_feats = self.features_str(feature_matrix[i])
            feat_file.write(str_feats)
            feat_file.write(" " + realized_tags[i])
            feat_file.write("\n")
        feat_file.flush() #make sure it is all written out
        #run java trainer
        print "Calling java MaxEntModelTrain to train classifier, storing to %s" % model_path
        feat_file.seek(0)
        failure = subprocess.call(['java', '-cp', self.OPENNLPCLASSPATH, 'MaxEntModelTrain', feat_file.name, model_path])
        if not failure:
            self.model_path = model_path
        feat_file.close()
        
    
    def read_results(self, file_handle):
        file_handle.seek(0)
        corpus_results = []
        sent_results = []
        for line in file_handle:
            tag = line.rstrip('\n')
            if len(tag) > 0:
                sent_results.append(tag)
            else:
                #sentences are separated by blank line
                corpus_results.append(sent_results)
                sent_results = []
        return corpus_results
    
        
    def tag_corpus(self, corpus, method = "greedy"):
        if self.model_path == None:
            raise ValueError("Missing a model path, train!")
            
        feat_file = tempfile.NamedTemporaryFile(mode="r+w")
        for sent in corpus:
            fake_tags = [ None ] * len(sent)
            sent_features = self.build_features_sent(zip(sent, fake_tags))
            for word_features in sent_features:
                str_feats = self.features_str(word_features)
                feat_file.write(str_feats)
                feat_file.write("\n") #new line for each word
            feat_file.write("\n") #blank line after every sentence
        feat_file.flush() #make sure it is all written out
        result_file = tempfile.NamedTemporaryFile(mode="r+w")
        print "Calling java MaxEntModelPredict to tag sentence"
        failure = subprocess.call(['java', '-cp', self.OPENNLPCLASSPATH, 'MaxEntModelPredict', self.model_path, feat_file.name, result_file.name, "-" + method])
        results = []
        if not failure:
            corpus_tags = self.read_results(result_file)
            for i in xrange(len(corpus)):
                results.append(zip(corpus[i], corpus_tags[i]))
            feat_file.close()
            result_file.close()
        return results

    