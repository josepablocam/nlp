#Author: Jose Pablo Cambronero
#Spring 2015 NLP Term Project
#jpc485@nyu.edu

#utilities
#relabel inv, oov, etc
relabel_vocab <- function(x) {
                if(x == "inv") "in vocabulary" 
                else if(x == "oov") "out of vocabulary"
                 else "aggregate"}


#Visualizing development set results

library(plyr)
library(ggplot2)
setwd("/Users/josecambronero/MS/S15/nlp/term_project/results")
devacc <- read.csv("dev_acc.csv")
devacc <- devacc[order(devacc$acc),]
devacc$cohort_name <- sapply(devacc$cohort, relabel_vocab)

dev_acc_graph <- ggplot(devacc, aes(x = modelcat, y = acc)) + geom_boxplot() + facet_wrap( ~cohort_name, scales ="free_y") +
                  labs(x = "Model category", y = "Tag accuracy on development set")

ggsave(filename = "dev_acc_box_whiskers.png", plot = dev_acc_graph)



#Visualizing test set results
#inv, oov, and aggregate test accuracy
testacc <- read.csv("test_acc.csv")
testacc$cohort_name <- sapply(testacc$cohort, relabel_vocab)
test_acc_graph <- ggplot(testacc, aes(x = modelcat, y = acc)) + 
                   geom_bar(stat="identity", position="dodge", aes(group = cohort_name, fill=cohort_name)) +
                   scale_y_continuous(breaks = seq(0.65, 1, by = 0.025) ) +
                   coord_cartesian(ylim = c(0.65, 1)) +
                   labs(y = "Tag accuracy on test set", x = "model", fill = "cohort") +
                   theme(legend.position="bottom") 
    
ggsave(filename = "test_acc_bars.png", plot = test_acc_graph)


#test accuracy by tag
testaccbytag <- read.delim("test_acc_by_tag.tsv")
maxent_tag_accuracy <- ggplot(subset(testaccbytag, grepl("maxent", model)), aes(x =  tagpct, y = acc)) + geom_point() + 
                          scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
                          scale_x_continuous(breaks = seq(0, 1, by = 0.01)) +
                          labs(y = "Tag accuracy in test set", x = "Tag concentration in training set") +
                          theme(axis.text.x = element_text(angle = 90))
ggsave(filename = "maxent_tag_accuracy_dot.png", plot = maxent_tag_accuracy)

#confusion matrix for maxent
confusion <- read.delim("confusion_table.tsv")
ggplot(confusion,aes(x = reorder(observed, observed), y = reorder(predicted, predicted))) + geom_tile(aes(fill = acc))

