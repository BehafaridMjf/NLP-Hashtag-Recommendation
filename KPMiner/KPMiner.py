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


import pke

key_list = []

for text in u_df['text'] :
        # 1. create a KPMiner extractor.
        extractor = pke.unsupervised.KPMiner()
        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en',
                                normalization=True)
        # 3. select {1-5}-grams that do not contain punctuation marks or
        #    stopwords as keyphrase candidates. Set the least allowable seen
        #    frequency to 5 and the number of words after which candidates are
        #    filtered out to 200.
        lasf = 5
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)
        # 4. weight the candidates using KPMiner weighting function.
        df = pke.load_document_frequency_file(input_file='/content/output.tsv.gz')
        alpha = 2.3
        sigma = 3.0
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)
        # 5. get the 10-highest scored candidates as keyphrases
        keywords = extractor.get_n_best(n=5)
        keywords = [w[0] for w in keywords]
        output = []
        for sentence in keywords:
            output.append(" ".join([ps.stem(i) for i in sentence.split()]))

        key_list.append(output)

u_df['KPMiner@5'] = key_list
