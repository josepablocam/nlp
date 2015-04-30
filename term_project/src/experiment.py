#Author: Jose Cambronero (N17381190)
#jpc485@nyu.edu
#Spring 2015 NLP Term Project

################ Utilities ###############################################################
from tigerutil import *
import os
import random
import re

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
     return re.search(r'maxent', name) != None 

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
print "Separating data into training and testing corpus"
trainpct = 0.7
devpct = 0.15
testpct = 0.15

#Training set
trainix = int(len(doc) * trainpct)
traindat = doc[:trainix] # we train on this

#Development set
devix = trainix + int(len(doc) * devpct)
devdat = doc[trainix:devix]
devdat_no_tags = rem_tags(devdat)

#Test set
testdat = doc[devix: ] #we report this
testdat_no_tags = rem_tags(testdat)




################################ HMM handling of unknown words ###########################
#Strategy 1: simple UNKNOWN
def simple_unknown(x):
    return 'UNKNOWN'

#Strategy 2: smarter, morphologically based unknown word tagger
#given knowledge of german and suffix distribution analysis
#see datadist.q
noun_regex = r'(eit|eit|ung|onie)$' #at end
verb_regex = r'en$' #at end
adj_regex = r'(ich|isch|ig)' #note not necessarily at end, deklination
num_regex = r'[0-9]'

def  morpho_unknown(word):
     if re.search(noun_regex, word):
         return "NN"
     elif re.search(verb_regex, word):
         return "VVINF"
     elif re.search(adj_regex, word):
         return "ADJA" #picked purely because it is most common vs ADJD in training data
     elif re.search(num_regex, word):
         return "CARD"
     else:
         return "UNKNOWN"


################################ Maxent feature sets ###########################
#All our maxent models include by default the previous 2 tags as features, 
#so any features listed below are in addition to those





################################ Model Creation ##########################################

##We create a dictionary that keeps an instance of each of our models
##We train each of them separately (which might be a bit inefficient, but simple)
print "Creating models"
models = {}
#bigram models using unknown word strategy 1
models["bigram kn s1 3"] = BigrammodelKN(labels, simple_unknown, 3)
models["bigram kn s1 6"] = BigrammodelKN(labels, simple_unknown, 6)
models["bigram lp s1 3"] = Bigrammodel(labels, simple_unknown, 3)
models["bigram lp s1 6"] = Bigrammodel(labels, simple_unknown, 6)
#bigram models using unknown word strategy 2
models["bigram kn s2 3"] = BigrammodelKN(labels, morpho_unknown, 3)
models["bigram kn s2 6"] = BigrammodelKN(labels, morpho_unknown, 6)
models["bigram lp s2 3"] = Bigrammodel(labels, morpho_unknown, 3)
models["bigram lp s2 6"] = Bigrammodel(labels, morpho_unknown, 6)
#trigram models using unknown word strategy 1
models["trigram lp s1 3"] = Trigrammodel(labels, simple_unknown, 3)
models["trigram lp s1 6"] = Trigrammodel(labels, simple_unknown, 6)
#trigram models using unknown word strategy 2
models["trigram lp s2 3"] = Trigrammodel(labels, morpho_unknown, 3)
models["trigram lp s2 6"] = Trigrammodel(labels, morpho_unknown, 6)
#maxent model feature set 1 
models["maxent f1"] = Maxentmodel(labels, feat_set1)
#maxent model feature set 2
models["maxent f2"] = Maxentmodel(labels, feat_set2)
#maxent model feature set 3
models["maxent f3"] = Maxentmodel(labels, feat_st3)


################ Model Training ##########################################################
##train the models, we can afford to retrain the maxentmodel whenever we want, since the underlying implementation 
#calls a fast java trainer
def train_models(models):
    for name, model in models.iteritems():
        print "Training %s" % name
        if is_maxent(name):
            model_path = str_to_path(name, " ", ".txt")
            model.train(devdat, model_path)
        else:
            model.train(devdat)
    

train_models(models)

################ Tag development set######################################################
BEAM_RANGE = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
def test_models(models, tagged_corpus, result_path):
    results = { }
    untagged_corpus = rem_tags(tagged_corpus)
    #run models
    for name, model in models.iteritems():
        print "Tagging with %s" % name
        if is_maxent(name): #for maxent models try both greedy and beam with low threshold
            results[name + " greedy"] = model.tag_corpus(untagged_corpus, method = "greedy")
            for beam in BEAM_RANGE:
                results[name + " beam"+beam] = model.tag_corpus(untagged_corpus, method = "viterbi", beam = beam)
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


test_models(models, devdat, "../results/dev_results.tsv")


################ Test set ################################################################
#Based on results from the development set, we pick the best models in each class and
best_models_names = []
best_models = {}
for name in best_models_names:
    best_models[name] = models[name]

test_models(best_models, testdat, "../results/test_results.tsv")






