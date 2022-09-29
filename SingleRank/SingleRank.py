
import pke

# define the set of valid Part-of-Speeches
# pos = {'NOUN', 'ADJ'}

key_list = []
# pos = {'NOUN','ADJ'}

for text in df['text']:

  # 1. create a SingleRank extractor.
  extractor = pke.unsupervised.SingleRank()
  # 2. load the content of the document.
  extractor.load_document(input= text,language='en',normalization=True)
  # 3. select the longest sequences of nouns and adjectives as candidates.
  extractor.candidate_selection()
  # 4. weight the candidates using the sum of their word's scores that are
  #    computed using random walk. In the graph, nodes are words of
  #    certain part-of-speech (nouns and adjectives) that are connected if
  #    they occur in a window of 10 words.

  extractor.candidate_weighting(window=10)
  # 5. get the 10-highest scored candidates as keyphrases
  keyword = extractor.get_n_best(n=5)
  keyword = [w[0] for w in keyword]

  output = []
  for sentence in keyword:
    output.append(" ".join([ps.stem(i) for i in sentence.split()]))
  key_list.append(output)

df['SingleRank@5'] = key_list
