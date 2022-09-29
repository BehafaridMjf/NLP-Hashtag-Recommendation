
import pke

# 1. create a YAKE extractor.
key_list1 = []
# key_list2 = []

for text in df['nj_text']:

  extractor = pke.unsupervised.YAKE()
  # 2. load the content of the document.

  extractor.load_document(input= text,
                          language='en',
                          normalization=True)
  # 3. select {1-3}-grams not containing punctuation marks and not
  #    beginning/ending with a stopword as candidates.
  extractor.candidate_selection(n=1)
  # 4. weight the candidates using YAKE weighting scheme, a window (in
  #    words) for computing left/right contexts can be specified.
  window = 2
  use_stems = False # use stems instead of words for weighting
  extractor.candidate_weighting(window=window,
                                use_stems=use_stems)
  # 5. get the 10-highest scored candidates as keyphrases.
  #    redundant keyphrases are removed from the output using levenshtein
  #    distance and a threshold.
  threshold = 0.8
  keyword = extractor.get_n_best(n= 5, threshold=threshold)
  key_list1.append(keyword)
  
  pip install wordninja
  
  import wordninja
key_lists = []
for i in df['Yake@5']:
  out = []
  for x in i:
    out.append(" ".join(wordninja.split(x)))
  key_lists.append(out)

key_lists

output = []
  for sentence in keyword:
    output.append(" ".join([ps.stem(i) for i in sentence.split()]))
  key_list.append(output)
  
  keyl = []
for i in key_list1:
  keyl.append([w[0] for w in i])
  
  
