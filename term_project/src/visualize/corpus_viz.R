#Author: Jose Pablo Cambronero
#Spring 2015 NLP Term Project
#jpc485@nyu.edu

setwd("/Users/josecambronero/MS/S15/nlp/term_project/results")
library(ggplot2)

#Distribution of tags in TIGER corpus
tag_dist <- read.delim("tagdist.tsv")
tag_dist <- tag_dist[order(tag_dist$pct, decreasing = TRUE), ]
top_10_tag_dist <- ggplot(subset(tag_dist, pct >= sort(pct, decreasing = T)[10]), aes(x = reorder(tag, pct), y = pct)) + 
                    geom_point() + theme(axis.text.x = element_text(angle = 90)) +
                    labs(x = "top 10 tags by frequency", y = "% of TIGER tags")
ggsave(filename = "tiger_tag_dist.png", top_10_tag_dist)

#Distribution of unknown words depending on minimum frequency count
unknown_dist <- read.csv("unknown_by_threshold.csv")
unknown_dist_graph <- ggplot(unknown_dist, aes(x = factor(threshold), y = unknown_pct)) +
  geom_point() + labs(y = "% of OOV tokens in test and development data", x = "Minimum token frequency")
ggsave(filename = "unknown_dist_graph.png", unknown_dist_graph)


