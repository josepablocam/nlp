//Some distributional stats on the training portion of our 
datpath:`:/Users/josecambronero/MS/S15/nlp/term_project/data/corpus/tiger_release_july03.tsv
n:28000 //the first n sentences are our training data, so only explore those
rawdat:{x where n>s:sums 0=count each x}read0 datpath //take only that data
//make it a table
tbl:flip `token`tag`sent!(`;`;"I")$'flip "\t"vs/:@[;where not w] rawdat,'"\t",/:string sums w:0=count each rawdat

//distribution of tags in training set
`pct xdesc update pct:ct%sum ct from select ct:count i by tag from tbl
//unsurprising to see noun

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
nsuffixrep:10 //we need to see the suffix at least 20 times
ndifftoken:20 //we want to see this suffix in at least 50 different words

suffixtbl:select from suffixtbl where nsuffixrep<(count;i) fby ([]suffixid;suffix), ndifftoken<(count distinct@;token) fby ([]suffixid;suffix)

//compare suffixes
suffixcomp:update pct:ct%sum ct by suffixid, suffix from select ct:count i by suffixid, suffix, tag from suffixtbl
//Use this to find endings that are very good at distinguishing
select from suffixcomp where pct>0.9




