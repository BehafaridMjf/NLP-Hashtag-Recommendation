for i in range(len(df)):

  with open(f'{i}.txt', 'w') as f:
      f.write(df['text'][i])
      
u_df = df.copy()


from pke import compute_document_frequency
from string import punctuation


"""Compute Document Frequency (DF) counts from a collection of documents.

N-grams up to 3-grams are extracted and converted to their n-stems forms.
Those containing a token that occurs in a stoplist are filtered out.
Output file is in compressed (gzip) tab-separated-values format (tsv.gz).
"""

# stoplist for filtering n-grams
stoplist=list(punctuation)

# compute df counts and store as n-stem -> weight values
compute_document_frequency(input_dir='/content/CN_txt',
                           output_file='/content/output.tsv.gz',
                           extension='xml',           # input file extension
                           language='en',                # language of files
                           normalization="stemming",    # use porter stemmer
                           stoplist=stoplist)

idf = pke.load_document_frequency_file(input_file='/content/output.tsv.gz')

import string
import pke

key_list = []
for text in u_df['text']:

  # 1. create a TfIdf extractor.
  extractor = pke.unsupervised.TfIdf()
  # 2. load the content of the document.

  extractor.load_document(input=text,
                          language='en',
                          normalization=None)
  # 3. select {1-3}-grams not containing punctuation marks as candidates.
  extractor.candidate_selection()
  # 4. weight the candidates using a `tf` x `idf`
  idf = pke.load_document_frequency_file(input_file='/content/output.tsv.gz')
  extractor.candidate_weighting(df=idf)
  # 5. get the 10-highest scored candidates as keyphrases
  keyword = extractor.get_n_best(n=5)
  keyword = [w[0] for w in keyword]

  output = []
  for sentence in keyword:
    output.append(" ".join([ps.stem(i) for i in sentence.split()]))
  key_list.append(output)
u_df['TfIdf@5'] = key_list
