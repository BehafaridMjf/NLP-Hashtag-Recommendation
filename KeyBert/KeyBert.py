!pip install keybert
from keybert import KeyBERT
kw_model = KeyBERT(model='all-mpnet-base-v2')


bert_list = []

for text in df['text']: 

  # keywords = kw_model.extract_keywords(text, 

  #                                    keyphrase_ngram_range=(0, 2), 

  #                                    stop_words='english', 

  #                                    highlight=False,

  #                                    top_n=15)
  

  keywords = kw_model.extract_keywords(text,keyphrase_ngram_range=(0, 2),stop_words='english',highlight=False,top_n=5)

  keywords = [w[0] for w in keywords]
  output = []
  for sentence in keywords:
      output.append(" ".join([ps.stem(i) for i in sentence.split()]))

  bert_list.append(output)

df['KeyBert@5'] = bert_list
