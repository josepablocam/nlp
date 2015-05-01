#Author: Jose Cambronero (N17381190)
#jpc485@nyu.edu
#Spring 2015 NLP Term Project

################ Utilities ###############################################################
from tigerutil import *
import os
import random
import re
import collections

##additional utilities useful for experiments
def rem_tags(corpus):    
    return [[word for word,tag in sent]for sent in corpus]

def accuracy(confusion_matrix):
    right = 0.0
    total = 0.0
    for (k1,k2),n in confusion_matrix.iteritems():
        right += n if k1 == k2 else 0
        total += n
    return right / total

    
#create a path from a string
def str_to_path(nm, sep, ext):
    return "_".join([e for e in nm.split(sep) if len(e) > 0]) + ext
   

#identify if a model name is associated with a maximum entropy model (these require some additional parameters, to store model and decode)
def is_maxent(model_name):
     return re.search(r'maxent', model_name) != None 

################ Model Implementations ###################################################
from bigrammodel import * #2-gram laplace smoothing
from bigrammodelkn import *  #2-gram KN smoothing
from trigrammodel import * #3-gram laplace smoothing
from maxentmodel import * #maximum entropy model


################ Tiger Corpus Data ######################################################
print "Reading in the tiger corpus"
doc = tigertsv_to_list("../data/corpus/tiger_release_july03.tsv")
random.seed(100)  # we pick a deterministic seed for reproducibility
random.shuffle(doc) ##shuffle our data
labels = list(set([tag for sent in doc for word,tag in sent])) #extract labels in all data

##Our training and testing data split parameters
print "Separating data into training, development and testing corpus"
trainpct = 0.7
devpct = 0.15
testpct = 0.15

#Training set
trainix = int(len(doc) * trainpct)
traindat = doc[:trainix] # we train on this

#Development set
devix = trainix + int(len(doc) * devpct)
devdat = doc[trainix:devix]

#Test set
testdat = doc[devix: ]




################################ HMM handling of unknown words ###########################
#Strategy 1: simple UNKNOWN
def simple_unknown(x):
    return 'UNKNOWN'


#Strategy 2: smarter, morphologically based unknown word tagger
#given knowledge of german and suffix distribution analysis
#see datadist.q
noun_regex = r'(eit|ung|onie|aft|ion|ik)$' #at end
verb_regex = r'en$' #at end
adj_regex = r'(ich|ig|eis|sch)' #note not necessarily at end, deklination
num_regex = r'[0-9]'
adv_regex = r'(so|auch)$'


def morpho_unknown(word):
     if re.search(noun_regex, word) and word[0].isupper(): #nouns in german are caps
         return "POSS_NOUN"
     elif re.search(verb_regex, word):
         return "POSS_VERB"
     elif re.search(adj_regex, word):
         return "POSS_ADJ" #picked purely because it is most common vs ADJD in training data
     elif re.search(num_regex, word):
         return "POSS_CARD"
     elif re.search(adv_regex, word) and not word[0].isupper():
         return 'POSS_ADV'
     else:
         return "UNKNOWN"


################################ Maxent feature sets ###########################
#All our maxent models include by default the previous 2 tags as features, 
#so any features listed below are in addition to those
def feat_set1():
    feats = {}
    feats["word"] = lambda word: word
    feats["suffix-3"] = lambda word: word[-3:]
    feats["suffix-2"] = lambda word: word[-2:]
    feats["last-letter"] = lambda word: word[-1]
    feats["firstCaps"] = lambda word: word[0].isupper()
    feats["has-number"] = lambda word: re.search(r'[0-9]', word) != None
    feats["is-long-word"] = lambda word: len(word) > 6
    return feats
 
#removes last-letter feature and adds prefixes, all caps, and hyphen feature
def feat_set2():
    feats = feat_set1()
    feats.pop("last-letter")
    feats["prefix-3"] = lambda word: word[:3]
    feats["prefix-2"] = lambda word: word[:2]
    feats["allCaps"] = lambda word: word.upper() == word
    feats["has-hyphen"] = lambda word: re.search(r'-', word) != None
    return feats   

    
################################ Model Creation ##########################################

##We create a dictionary that keeps an instance of each of our models
##We train each of them separately (which might be a bit inefficient, but simple)
print "Creating models"
models = {}
#bigram models using unknown word strategy 1
#delta 0f 0.75 for kneser ney discount
models["bigram kn s1 3 d 0.75"] = BigrammodelKN(labels, simple_unknown, 3, delta = 0.75)
models["bigram kn s1 6 d 0.75"] = BigrammodelKN(labels, simple_unknown, 6, delta = 0.75)
models["bigram kn s1 3 d 0.5"] = BigrammodelKN(labels, simple_unknown, 3, delta = 0.5)
models["bigram kn s1 6 d 0.5"] = BigrammodelKN(labels, simple_unknown, 6, delta = 0.5)
models["bigram lp s1 3"] = Bigrammodel(labels, simple_unknown, 3)
models["bigram lp s1 6"] = Bigrammodel(labels, simple_unknown, 6)
#bigram models using unknown word strategy 2
models["bigram kn s2 3 d 0.75"] = BigrammodelKN(labels, morpho_unknown, 3, delta = 0.75)
models["bigram kn s2 6 d 0.75"] = BigrammodelKN(labels, morpho_unknown, 6, delta = 0.75)
models["bigram kn s2 3 d 0.5"] = BigrammodelKN(labels, morpho_unknown, 3, delta = 0.5)
models["bigram kn s2 6 d 0.5"] = BigrammodelKN(labels, morpho_unknown, 6, delta = 0.5)
models["bigram lp s2 3"] = Bigrammodel(labels, morpho_unknown, 3)
models["bigram lp s2 6"] = Bigrammodel(labels, morpho_unknown, 6)
#trigram models using unknown word strategy 1
models["trigram lp s1 3"] = Trigrammodel(labels, simple_unknown, 3)
models["trigram lp s1 6"] = Trigrammodel(labels, simple_unknown, 6)
#trigram models using unknown word strategy 2
models["trigram lp s2 3"] = Trigrammodel(labels, morpho_unknown, 3)
models["trigram lp s2 6"] = Trigrammodel(labels, morpho_unknown, 6)
#maxent model feature set 1 
models["maxent f1"] = Maxentmodel(labels, feat_set1())
#maxent model feature set 2
models["maxent f2"] = Maxentmodel(labels, feat_set2())



################ Model Training ##########################################################
##train the models, we can afford to retrain the maxentmodel whenever we want, since the underlying implementation 
#calls a fast java trainer
def train_models(models, tagged_corpus):
    for name, model in models.iteritems():
        print "Training %s" % name
        if is_maxent(name):
            model_path = str_to_path(name, " ", ".txt")
            model.train(tagged_corpus, model_path)
        else:
            model.train(tagged_corpus)
    

train_models(models, traindat)

################ Tag development set######################################################
def test_models(models, tagged_corpus, beam_range, result_path):
    results = { }
    untagged_corpus = rem_tags(tagged_corpus)
    #run models
    for name, model in models.iteritems():
        print "Tagging with %s" % name
        if is_maxent(name): #for maxent models try both greedy and beam with low threshold
            results[name + " greedy"] = model.tag_corpus(untagged_corpus, method = "greedy")
            for beam in beam_range:
                print "with beam %f" % beam
                results[name + " beam " + str(beam)] = model.tag_corpus(untagged_corpus, method = "viterbi", beam = beam)
        else:
            results[name] = model.tag_corpus(untagged_corpus)   
    #write out results
    results_file = open(result_path, "w")
    results_file.write("model\tobserved\tpredicted\tfreq\n")
    calc_matrix = Ngrammodel(labels, lambda x: x).confusion_matrix_corpus #just a dummy to calculate matrix
    for name, result in results.iteritems():
        clean_model_name = "_".join([e for e in name.split(" ") if len(e) > 0])
        confusion_matrix = calc_matrix(tagged_corpus, result)
        for (observed, predicted), cts in confusion_matrix.iteritems():
            results_file.write("%s\t%s\t%s\t%f\n" % (clean_model_name, observed, predicted, cts))
        print "%s Accuracy: %f" % (name, accuracy(confusion_matrix))
    results_file.close()
    return results


#confusion matrix broken out by oov status
def confusion_matrix_by_oov(models, model_results, tagged_corpus, train_corpus, result_path):
    untagged_train = rem_tags(train_corpus)
    train_words = set([word for sent in untagged_train for word in sent])
    results = collections.defaultdict(int)
    corp_len = len(tagged_corpus)
    sent_len = [len(sent) for sent in tagged_corpus]
    
    for name, predictions in model_results.iteritems():
        if is_maxent(name):
            ref_words = train_words
        else:
            ref_words = models[name].freq_words
        print "calculating accuracy by tag and oov status for %s" % name
        clean_model_name = "_".join([e for e in name.split(" ") if len(e) > 0])
        for sent_i in xrange(corp_len):
            obs_sent = tagged_corpus[sent_i]
            pred_sent = predictions[sent_i]
            for word_i in xrange(sent_len[sent_i]):
                word, observed = obs_sent[word_i]
                word, predicted = pred_sent[word_i]
                is_oov = not word in ref_words
                results[(clean_model_name, observed, predicted, is_oov)] += 1
    print "writing out results to %s" % result_path
    results_file = open(result_path, "w")
    results_file.write("model\tobserved\tpredicted\voov\freq\n")
    for (model_name, observed, predicted, is_oov),freq in results.iteritems():
        results_file.write("%s\t%s\t%s\t%d\t%d\n") % (model_name, observed, predicted, 1 if is_oov else 0, freq)
    return results
            
        

#this takes a long time....go grab coffee :)
dev_beams = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
dev_results = test_models(models, devdat, dev_beams,  "../results/dev_results.tsv")
dev_oov = oov_calc(models, dev_results, devdat, traindat, ".../results/dev_oov.tsv")

################ Tag test set ############################################################
#Based on results from the development set, we pick the best models in each class and
#best_models_names = ["bigram kn s2 3", "trigram lp s2 3", "maxent f2"]
#best_models = {}
#for name in best_models_names:
#    best_models[name] = models[name]

#opt_beam = [ 0.5 ]
#test_results = test_models(best_models, testdat, opt_beam, "../results/test_results.tsv")
#test_oov = oov_calc(best_models, traindat, testdat, ".../results/test_oov.tsv")





