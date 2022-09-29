import pke

key_list = []

for text in df['text']:
        # define the valid Part-of-Speeches to occur in the graph
        # pos = {'NOUN', 'ADJ'}
        # define the grammar for selecting the keyphrase candidates
        # grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        # 1. create a PositionRank extractor.
        extractor = pke.unsupervised.PositionRank()
        # 2. load the content of the document.
        extractor.load_document(input= text,
                                language='en',
                                normalization= True)
        # 3. select the noun phrases up to 3 words as keyphrase candidates.
        #grammar here
        extractor.candidate_selection()
        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    10 words.
        extractor.candidate_weighting(window=10,
                                      pos= None)
        # 5. get the 10-highest scored candidates as keyphrases
        keyword = extractor.get_n_best(n=5)
        keyword = [w[0] for w in keyword]

        output = []
        for sentence in keyword:
          output.append(" ".join([ps.stem(i) for i in sentence.split()]))
        key_list.append(output)

df['PositionRank@5'] = key_list
