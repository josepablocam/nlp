#Author: Jose Cambronero (N17381190)
#jpc485@nyu.edu
#Spring 2015 NLP Term Project

from os import path
import codecs

def tigerformat_to_tsv(input_name, output_name):
    collected = []
    in_sentence = False
    sep = "\t"
    quote_marks = ["``", "''"]
    
    input = codecs.open(input_name, "r", encoding="latin1")

    for line in input:
        if in_sentence:
            if line[0] == "#":
                in_sentence = False
                collected.append("\n")
            else:
                word,tag = filter(lambda x: x != "", line.split(sep))[:2]
                if word in quote_marks:
                    word = "\""
                collected.append(sep.join((word, tag)) + "\n")
        else:
            if line.split(" ")[0] == "#BOS":
                in_sentence = True
            
               
    output = codecs.open(output_name, "w", encoding="latin1")
    output.writelines(collected)
         
    input.close()
    output.close()



def tigertsv_to_list(input_name):
    doc = []
    sent = []
    sep = "\t"
    
    input = codecs.open(input_name, "r", encoding="latin1")
    
    for line in input:
        if line == "\n":
            doc.append(sent) #append sentence to doc
            sent = [] #reinitialize sentence to empty
        else:
            word,tag = line.rstrip('\n').split(sep)
            sent.append((word, tag)) #append new tuple
    
    return doc
