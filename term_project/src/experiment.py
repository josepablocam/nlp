from tigerutil import *
from bigrammodel import *
from trigrammodel import *
import random
import re


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



###our data
doc = tigertsv_to_list("/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv")
##set our seed
random.seed(100)  # we pick a deterministic seed for reproducibility
##shuffle our data
random.shuffle(doc)
labels = list(set([tag for sent in doc for word,tag in sent]))


##Our training and testing data split parameters
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

def smarter_suffix(word):
     if re.search(noun_regex, word):
         return "NN"
     elif re.search(verb_regex, word):
         return "VVINF"
     elif re.search(adj_regex, word):
         return "ADJA" #picked purely because it is most common vs ADJD in training data
     else:
         return "UNKNOWN"
    
models = {}
models["bigram simple unknown agg 6"] = Bigrammodel(labels, lambda x: "UNKNOWN", 6)
models["trigram simple unknown agg 6"] = Trigrammodel(labels, lambda x: "UNKNOWN", 6)
models["bigram simple morpho agg 6"] = Bigrammodel(labels, smarter_suffix, 6)
models["trigram simple morpho agg 6"] = Trigrammodel(labels, smarter_suffix, 6)
models["maxent model"] = Maxentmodel(labels)



##train the models, in case of maximum entropy try to read, if not train and save (takes forever to train)
for name, model in models.iteritems():
    if(name == "maxent model"):
        if os.path.isfile("maxent_model"):
            print "Reading pickled max entropy model"
            model[name].model = pickle.load("maxent_model")
        else:
            print "Training maximum entropy model and then pickling"
            model[name].train(devdat)
            file = open("maxent_model", "w")
            pick.dump(model[name].model, file)
            file.close()
    else:    
        print "Training %s" % name
        model.train(devdat)
    

###Now tag the test data and store results
testdat_no_tags = rem_tags(testdat)
results = { }

##this will take a while, go grab coffee....
for name, model in models.iteritems():
    "Tagging with %s" % name
    results[name] = model.tag_corpus(testdat_no_tags)

###write out confusion matrix as a simple csv table so we can analyze errors in q
###which is faster and easier with sql-like syntax
results_file = open("model_results.csv", "w")
results_file.write("model, observed, predicted,freq\n")
for name, result in results.iteritems():
    confusion_matrix = models[name].confusion_matrix_corpus(testdat, result)
    for observed, predicted in confusion_matrix:
        clean_name = "_".join([word for word in name.split(" ") if len(word) > 0])
        file.write("%s, %s, %s, %f\n") %(name, observed, predicted, confusion_matrix[(observed,predicted)])
    print "%s Accuracy: %f" % (name, accuracy(confusion_matrix))

results_file.close()





        
