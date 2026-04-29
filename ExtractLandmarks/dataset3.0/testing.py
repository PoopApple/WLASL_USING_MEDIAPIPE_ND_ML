# %%
import string
import os
import spacy
import wordninja

# %%
nlp = spacy.load("en_core_web_sm")

def split_compound(word):
    """Split a concatenated word into its component English words.
    e.g. 'absolutelynothing' -> ['absolutely', 'nothing']
    Single real words like 'basketball' are returned as-is.
    """
    parts = wordninja.split(word)
    return parts if parts else [word]

def lemmatize(word):
    """Lemmatize a (possibly compound) word.
    Splits into parts first, lemmatizes each, then joins with underscore.
    e.g. 'absolutelynothing' -> 'absolutely_nothing' (after lemma)
    """
    parts = split_compound(word)
    # Lemmatize each part individually so spaCy sees real words
    lemmatised_parts = []
    for part in parts:
        doc = nlp(part)
        lemmatised_parts.append(doc[0].lemma_)
    return "_".join(lemmatised_parts)

# %%
datasetpath = "./landmarks_npz"

# %%
print(len(os.listdir(datasetpath)))

# %%
wordlist = os.listdir(datasetpath)

og_clean_wordlist = {word : word.rstrip(string.digits).replace(".","").lower() for word in wordlist}

# print(wordlist)


# %%
print(len(og_clean_wordlist))

# %%
print(og_clean_wordlist['ABSOLUTELYNOTHING'])

# %%

lemmatisedwords = {og_word : lemmatize(clean_word) for og_word,clean_word in og_clean_wordlist.items()}

# lemmatisedwords = {word:lemmatize(word) for word in wordlist}
with open("./dataset3_words_lemmatised.txt","w") as f:
    for k,v in lemmatisedwords.items():
        f.write(f"{v}\n")
    
dataset_words_lemmatised = lemmatisedwords.values()

print(len(dataset_words_lemmatised))

# %%
#TEST

import spacy
nlp = spacy.load("en_core_web_sm")

word = "ABSOLUTELYNOTHING"
doc = nlp(word)

for token in doc:
    print(token.text, "->", token.lemma_)

# %%
print(lemmatisedwords['ABSOLUTELYNOTHING'])

# %%
set_of_dataset_words = set(dataset_words_lemmatised)

print(len(set_of_dataset_words))
print(len(dataset_words_lemmatised))

# %%
clean_oglist = {}

for og,lem in lemmatisedwords.items():
    if clean_oglist.get(lem):
        clean_oglist[lem].append(og)
    else:
        clean_oglist[lem] = [og]


for x,y in sorted(clean_oglist.items()):
    print(f"{x} => {",".join(y)}")


