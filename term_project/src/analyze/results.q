/
    Author: Jose Cambronero (jpc485@nyu.edu)
    NLP Term Project Spring 2015

    We analyze the results from our various POS tagging models
\

//utilities
calc_accs:{
 accs:0!select acc:sum[freq*observed=predicted]%sum freq by model, cohort:?[oov;`oov;`inv] from x;
 accs,0!select acc:sum[freq*observed=predicted]%sum freq by model, cohort:`all from x
 }

mapcat:{`$first each "_"vs/:string x}

/ ***** Development results ******* /
devrespath:`:/Users/josecambronero/MS/S15/nlp/term_project/results/dev_oov.tsv
devres:("SSSBF"; enlist "\t") 0:devrespath
devacc:calc_accs devres
devacc:update modelcat:mapcat model from devacc;
bestbycat:exec (exec distinct cohort from devacc)#cohort!acc by model:model from devacc where model in (exec model from devacc where cohort=`all, acc= (max;acc) fby modelcat)

`:/Users/josecambronero/MS/S15/nlp/term_project/results/dev_acc.csv 0:csv 0:devacc
`:/Users/josecambronero/MS/S15/nlp/term_project/results/dev_best_models.csv 0:csv 0:bestbycat

/ ***** Test results ******* /
testrespath:`:/Users/josecambronero/MS/S15/nlp/term_project/results/test_oov.tsv
testres:("SSSBF"; enlist "\t") 0:testrespath
testres:delete from testres where model like "*greedy*"
testacc:calc_accs testres
testacc:update modelcat:mapcat model from testacc

testaccbytag:select acc:sum[freq*predicted=observed]%sum freq by model, observed from testres
tagdisttrain:("SFF";enlist "\t") 0:`:/Users/josecambronero/MS/S15/nlp/term_project/results/tag_dist_training.tsv //concentration of tags in training data
testaccbytag:update tagpct:(exec tag!pct from tagdisttrain) observed from testaccbytag;



/* Maximum entropy confusion matrix */
worsttags:exec observed from testaccbytag where model like "maxent*", acc<0.85;
confusion:`observed xasc `predpct xdesc update predpct:freq%sum freq by observed from select sum freq by observed, predicted from testres where model like "maxent*", observed in worsttags
confusion:ungroup select 2 sublist predicted, 2 sublist predpct by observed from confusion where predicted<>observed
taginfo:1!flip `tag`info`ex!("SSS";"\t") 0:read0`:/Users/josecambronero/MS/S15/nlp/term_project/data/tag_description_clean.tsv
confusion:confusion lj `observed`obsinfo`obsex xcol taginfo
confusion:confusion lj `predicted`predinfo`predex xcol taginfo
testtagcon:update pct:freq%sum freq from select sum freq by observed from testres where model=first model //concentration of each tag in the test data set
confusion:update tagcon:(exec observed!pct from testtagcon) observed from confusion
confusion:update accuracy:(exec observed!acc from testaccbytag where model like "maxent*")observed from confusion
confusion:`tagcon xdesc `observed`predicted`predpct`tagcon`accuracy xcols confusion
confusion:@[confusion;exec c from meta confusion where t="s";{`$ssr'[string x;",";"/"]}]


`:/Users/josecambronero/MS/S15/nlp/term_project/results/test_acc.csv 0:csv 0:testacc
`:/Users/josecambronero/MS/S15/nlp/term_project/results/test_acc_by_tag.tsv 0:"\t" 0:testaccbytag
`:/Users/josecambronero/MS/S15/nlp/term_project/results/confusion_table.csv 0:csv 0:confusion





