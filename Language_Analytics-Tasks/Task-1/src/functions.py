import numpy as np

def clean_text(text):

    '''
    Cleans up the text by removing strings, that start with "<".

    Argument:
        Takes text object as an input.

    Returns:
        Returns the text without strings that are between "<...>" symbols.

    '''
    words = text.split()

    # The following variable will store only those tokens of text, that do not start with "<":
    clean_text = ""

    # Looping through words:
    for word in words:
        if not word[0] == "<":
            clean_text += word + " "
    
    return clean_text


def count_freq(doc):
    '''
    Calculates token`s relative frequency in a text as POS.

    Argument: 
        doc - text object, that was created using spaCy library, with a function nlp().
    
    Returns:
        The function returns a list of relative frequencies of interest. In this case, it returns a list containing relative
        frequency of NOUN, VERB, ADJ and ADV, respectively.
    '''
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0
    for token in doc:
        if token.pos_ == "NOUN":
            noun_count += 1
        elif token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "ADJ":
            adj_count += 1
        elif token.pos_ == "ADV":
            adv_count += 1

    # relative frequency per 10,000 words, rounded:

    rel_freq_noun = round((noun_count/len(doc)) * 10000, 2)
    rel_freq_verb = round((verb_count/len(doc)) * 10000, 2)
    rel_freq_adj = round((adj_count/len(doc)) * 10000, 2)
    rel_freq_adv = round((adv_count/len(doc)) * 10000, 2)

    return([rel_freq_noun, rel_freq_verb, rel_freq_adj, rel_freq_adv])

def unique_ent(doc):

    '''
    Calculates how many unique named entities of interest there are in a text file.

    Argument: 
        doc - text object, that was created using spaCy library, with a function nlp().
    
    Returns:
        Returns a list of values, which represent unique number of PERSONs, LOCs, ORGs, respectively, in a given text.
    '''
    ent_per = []
    ent_loc = []
    ent_org = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_per.append(ent.text)
        elif ent.label_ == "LOC":
            ent_loc.append(ent.text)
        elif ent.label_ == "ORG":
            ent_org.append(ent.text)
    
    count_per = len(np.unique(ent_per))
    count_loc = len(np.unique(ent_loc))
    count_org = len(np.unique(ent_org))
    
    return([count_per, count_loc, count_org])