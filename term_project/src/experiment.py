from tigerutil import *
from bigrammodel import *
import random
import re


###our data
doc = tigertsv_to_list("/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv")
##set our seed
random.seed(100)  # we pick a deterministic seed for reproducibility
##shuffle our data
random.shuffle(doc)

##Our training and testing data split parameters
trainpct = 0.7
trainix = int(len(doc) * trainpct)
##TODO: split up into development data and training data
devdat = doc[:trainix] # we train on this
testdat = doc[trainix: ] #we report this

##utilities
def rem_tags(corpus):    
    return [[word for word,tag in sent]for sent in corpus]

def accuracy(confusion_matrix):
    right = 0.0
    total = 0.0
    for (k1,k2),n in confusion_matrix.iteritems():
        right += n if k1 == k2 else 0
        total += n
    return right / total


#simple bigram models
models = {}
models["bigram simple unknown agg 2"] = Bigrammodel(lambda x: "UNKNOWN", 2)
models["bigram simple unknown agg 4"] = Bigrammodel(lambda x: "UNKNOWN", 4)
models["bigram simple unknown agg 6"] = Bigrammodel(lambda x: "UNKNOWN", 6)

#we define a smarter morphologically based unknown word tagger, given knowledge of german and suffix distribution analysis
## see datadist.q
noun_regex = r'(keit|heit|ung|onie)$' #at end
verb_regex = r'en$' #at end
adj_regex = r'(ich|isch|ig)' #note not necessarily at end, deklination

def smarter_suffix(word):
    if re.search(noun_regex, word):
        return "NN"
    elif re.search(verb_regex, word):
        return "VVINF"
    elif re.search(adj_regex, word):
        return "ADJA" #picked purely because it is most common vs ADJD in training data
    else:
        return "UNKNOWN"
    
#simple bigram models with smarter morphological tags for rare words
models["bigram simple morpho agg 2"] = Bigrammodel(smarter_suffix, 2)
models["bigram simple morpho agg 4"] = Bigrammodel(smarter_suffix, 4)
models["bigram simple morpho agg 6"] = Bigrammodel(smarter_suffix, 6)


#trigram models
#models["trigram simple unknown agg 2"] = Trigrammodel(lambda x: "UNKNOWN", 2)
#models["trigram simple unknown agg 4"] = Trigrammodel(lambda x: "UNKNOWN", 4)
#models["trigram simple unknown agg 6"] = Trigrammodel(lambda x: "UNKNOWN", 6)
#models["trigram simple morpho agg 2"] = Trigrammodel(lambda x: "UNKNOWN", 2)
#models["trigram simple morpho agg 4"] = Trigrammodel(lambda x: "UNKNOWN", 4)
#models["trigram simple morpho agg 6"] = Trigrammodel(lambda x: "UNKNOWN", 6)

#loglinear models
#models["trigram simple unknown agg 2, feature set 1"] = Loglinearmodel(lambda x: "UNKNOWN", 2)
#models["trigram simple unknown agg 4, feat set 1"] = Loglinearmodel(lambda x: "UNKNOWN", 4)
#models["trigram simple unknown agg 6, feat set 1"] = Loglinearmodel(lambda x: "UNKNOWN", 6)
#models["trigram simple morpho agg 2, feat set 2"] = Loglinearmodel(lambda x: "UNKNOWN", 2)
#models["trigram simple morpho agg 4, fet set 2"] = Loglinearmodel(lambda x: "UNKNOWN", 4)
#models["trigram simple morpho agg 6, feat set 2"] = Loglinearmodel(lambda x: "UNKNOWN", 6)




##train the models
for model in models:
    model.train(traindat)
    
###Now tag the test data and store results
test_dat_no_tags = rem_tags(test_dat)
results={}
##this will take a while, go grab coffee....
for name,model in models:
    results[name] = model.tag_corpus(test_dat_no_tags)

###write out the results for more detailed analysis
for name, result in results:
    ###write out confusion matrix as a simple csv table
    







#>>> accuracy(comp)
#0.9579889299994626 on training
#>>> accuracy(comp_new) on test data
#0.928444493146508
#>>a per tag accuracy/recall should tell us what needs to be improved




        
