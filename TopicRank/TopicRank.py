import pke

from pke.lang import stopwords


extractor = pke.unsupervised.YAKE()

stoplist = stopwords.get('english')
extractor.load_document(input='path',
                        language='en',
                        stoplist=stoplist,
                        normalization=None)

#n can be 1,2,3
extractor.candidate_selection(n=1)

#set the window length
window = 2
use_stems = False 

extractor.candidate_weighting(window=window,
                              use_stems=use_stems)
#levenshtein distance
threshold = 0.8

keyphrases = extractor.get_n_best(n=5, threshold=threshold)

