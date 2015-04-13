from tigerutil import *
doc = tigertsv_to_list("/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv")

from bigrammodel import *
m = Bigrammodel(lambda x: "UNKNOWN", 2)
m.train(doc)

doc_without_tags = [[word for word,tag in sent] for sent in doc]

import cProfile
test = doc_without_tags[:20000]
5238
def test():
    for i in xrange(len(doc_without_tags)):
        if(i % 1000 == 0):
            print "running %d" % i
        res = m.tag_sent(doc_without_tags[i])
        

cProfile.run('res = m.tag_corpus(test)')
cProfile.run('m.tag_sent_slow(doc_without_tags[10])')


predicted = m.tag_corpus(doc_without_tags)


