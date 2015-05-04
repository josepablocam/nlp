//Author: Jose Pablo Cambronero (jpc485@nyu.edu)
//NLP Spring 2015 Final project
//ID: N17381190

/
    We compute some distributional statistics for suffixes in the TIGER corpus, in order
    to identify those that might potentially help with unknown word classification
    We seek to find suffixes which generalize well (meaning multiple words have it)
    and that tend to be associated with a specific tag, so that our distribution is concentrated
    around that tag
\


//Some distributional stats on the training portion of our 
datpath:`:/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv

//make it a table
tbl:flip `token`tag`sent!(`;`;"I")$'flip "\t"vs/:@[;where not w] rawdat,'"\t",/:string sums w:0=count each rawdat:read0 datpath

//distribution of tags in corpus, note the large concentration of nouns (unsurprising)
tagdist:`pct xdesc update pct:ct%sum ct from select ct:count i by tag from tbl

//Baseline:if we guess the most common tag by word and classify unknown according to tagdistribtion?
traindat:28000#tbl;
testdat:28000_tbl;
minct:3;
mle:exec {c?max c:count each group x}tag by token from update token:`UNKNOWN from traindat where minct>(count;i) fby token

//Accuracy on test data if we only use most common tag per word
exec acc:avg tag=mle[`UNKNOWN]^mle token from testdat
select acc:avg tag=mle[`UNKNOWN]^mle token by oov:not token in key mle from testdat



//unknown word distribution by mininmum frequency in training data 
unknowndist:flip `threshold`unknown_pct!(mincts; {avg not testdat[`token] in exec distinct token from traindat where x<=(count;i) fby token} each mincts:1+til 20)




//for tokens that repeat, we want to see distribution of tags within token
`ct`pct xdesc update pct:ct%sum ct by token from select ct:count i by token,tag from tbl where 1<(count@;i) fby token
//Some of these are pretty much guaranteed to be that tag, consider the following cutoff

//trying to do some work with suffixes
//we try 4 different lengths of suffixes, that encompass a lot of german endings:
//heit, keit, lich, isch, ung, en, weise, schaft
//we also inform ourselves with http://en.wiktionary.org/wiki/Category:German_suffixes
slen:2 3 4
suffixtbl:select from tbl where min[slen]<count each string token //only want terms longer than min suffix
suffixtbl:update suffixid:distinct each (count each string token)&count[i]#enlist slen from tbl  //cap length of suffix at length of token
suffixtbl:update suffix:neg[suffixid] sublist'string token from ungroup suffixtbl //calc suffix
suffixtbl:update suffixhexa:"x"$string suffix from suffixtbl //store suffix as hex to deal with foreign chars
suffixtbl:delete from suffixtbl where suffixid=1 //too short to be useful


//We're interested in suffixes with at least some frequency, and 
//that generalize across different words
//so let's focus only on suffixes that hit those requirements
nsuffixrep:10 //we need to see the suffix at least 10 times
ndifftoken:20 //we want to see this suffix in at least 20 different words

suffixtbl:select from suffixtbl where nsuffixrep<(count;i) fby ([]suffixid;suffix), ndifftoken<(count distinct@;token) fby ([]suffixid;suffix)

//compare suffixes
suffixcomp:update pct:ct%sum ct by suffixid, suffix from select ct:count i by suffixid, suffix, tag from suffixtbl
//Use this to find endings that are very good at distinguishing
select from suffixcomp where pct>0.9







hsym[`$"../results/tagdist.tsv"] 0:"\t" 0:tagdist
show "\t" 0:tagdist

hsym[`$"../results/unknown_by_threshold.csv"] 0:csv 0:unknowndist
