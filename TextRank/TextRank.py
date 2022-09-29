import pke

key_list = []
# pos = {'NOUN', 'ADJ'}
for text in df['text']:
    # define the set of valid Part-of-Speeches

    # 1. create a TextRank extractor.
    extractor = pke.unsupervised.TextRank()
    # 2. load the content of the document.
    extractor.load_document(input= text,
                            language='en',
                            normalization= True)
    # 3. build the graph representation of the document and rank the words.
    #    Keyphrase candidates are composed from the 33-percent
    #    highest-ranked words.
    extractor.candidate_weighting(window=2,
                                  top_percent=0.33)
    # 4. get the 10-highest scored candidates as keyphrases
    keyword = extractor.get_n_best(n=5)
    keyword = [w[0] for w in keyword]

    output = []
    for sentence in keyword:
      output.append(" ".join([ps.stem(i) for i in sentence.split()]))
    key_list.append(output)
df['TextRank@5'] = key_list
