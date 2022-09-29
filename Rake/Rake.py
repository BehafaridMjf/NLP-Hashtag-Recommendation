nltk.download('punkt')
pip install rake-nltk

from rake_nltk import Rake
r = Rake()

key_list = []
i= df['text'][1]
#Rake
r = Rake()

# r = Rake()
# r = Rake()
r.extract_keywords_from_text(i)
keyword = r.get_ranked_phrases_with_scores()
keyword = [w[1] for w in keyword]

output = []
for sentence in keyword:
    output.append(" ".join([ps.stem(i) for i in sentence.split()]))
    key_list.append(output)
output[0:2]
