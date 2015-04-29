#utilities
from tigerutil import *
import os
import random
import re

#model implementations
from bigrammodelkn import * 
from trigrammodel import *
from maxentmodel import *

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
    return "_".join([e for e in nm.split(sep) if len(e) > 0]) + "." + ext

#identify if a model name is associated with a maximum entropy model (these require some additional parameters, to store model and decode)
is_maxent = lambda x: re.search(r'maxent', name) != None

###our data
print "Reading in the tiger corpus"
doc = tigertsv_to_list("/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv")
##set our seed
random.seed(100)  # we pick a deterministic seed for reproducibility
##shuffle our data
random.shuffle(doc)
labels = list(set([tag for sent in doc for word,tag in sent]))

##Our training and testing data split parameters
print "Separating data into training and testing corpus"
trainpct = 0.7
trainix = int(len(doc) * trainpct)
devdat = doc[:trainix] # we train on this
testdat = doc[trainix: ] #we report this
testdat_no_tags = rem_tags(testdat)

#we define a smarter morphologically based unknown word tagger, given knowledge of german and suffix distribution analysis
#see datadist.q
noun_regex = r'(keit|heit|ung|onie)$' #at end
verb_regex = r'en$' #at end
adj_regex = r'(ich|isch|ig)' #note not necessarily at end, deklination
num_regex = r'[0-9]'

def smarter_suffix(word):
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


##We create a dictionary that keeps an instance of each of our models
print "Creating models"
models = {}
models["bigram kn simple unknown agg 3"] = BigrammodelKN(labels, lambda x: "UNKNOWN", 3)
models["trigram laplace simple unknown agg 3"] = Trigrammodel(labels, lambda x: "UNKNOWN", 3)
models["bigram kn simple morpho agg 3"] = BigrammodelKN(labels, smarter_suffix, 3)
models["trigram laplace simple morpho agg 3"] = Trigrammodel(labels, smarter_suffix, 3)
models["simple maxent model"] = Maxentmodel(labels, {'word' : lambda x: x}) #solely consider word and previous 2 tags (these are added by the class)
models["robust maxent model"] = Maxentmodel(labels) #use default features, which perform well



##train the models, we can afford to retrain the maxentmodel whenever we want, since the underlying implementation 
#calls a fast java trainer
for name, model in models.iteritems():
    print "Training %s" % name
    if is_maxent(name):
        model_path = str_to_path(name, " ", ".txt")
        model.train(devdat, model_path)
    else:
        model.train(devdat)
    
###Now tag the test data and store results
testdat_no_tags = rem_tags(testdat)
results = { }

##this will take a while, go grab coffee....
BEAM_FACTOR = 0.75
for name, model in models.iteritems():
    print "Tagging with %s" % name
    if  is_maxent(name): #for maxent models try both greedy and beam with low threshold
        results[name + " greedy"] = model.tag_corpus(testdat_no_tags, method = "greedy")
        results[name + " beam"] = model.tag_corpus(testdat_no_tags, method = "viterbi", beam = BEAM_FACTOR)
    else:
       results[name] = model.tag_corpus(testdat_no_tags)

###write out confusion matrix as a simple csv table so we can analyze errors in q
###which is faster and easier with sql-like syntax
results_file = open("model_results.csv", "w")
results_file.write("model, observed, predicted,freq\n")
calc_matrix = Ngrammodel(labels, lambda x: x).confusion_matrix_corpus #just a dummy to calculate matrix
for name, result in results.iteritems():
    clean_model_name = "_".join([e for e in name.split(" ") if len(e) > 0])
    confusion_matrix = calc_matrix(testdat, result)
    for (observed, predicted), cts in confusion_matrix.iteritems():
        results_file.write("%s, %s, %s, %f\n" % (clean_model_name, observed, predicted, cts))
    print "%s Accuracy: %f" % (name, accuracy(confusion_matrix))

results_file.close()


        

tags_by_model = defaultdict(set)
for model_name, model_results in results.iteritems():
    for sent in model_results:
        for word, tag in sent:
            tags_by_model[model_name].add(tag)
            
            
model_name = 'trigram laplace simple morpho agg 3'
explore = results[model_name]
for sent in explore:
    tags = [tag for word,tag in sent]
    if 'START' in tags:
        sent

    


