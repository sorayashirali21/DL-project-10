#The LDA model is also included in
#scikit-learn:
from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


groups = fetch_20newsgroups(subset='all', categories=categories)


from nltk.corpus import names
all_names = set(names.words())



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)



from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

#we use the term matrix defined previously as input to the LDA model:
data = count_vector.fit_transform(data_cleaned)


from sklearn.decomposition import LatentDirichletAllocation

t = 20
lda = LatentDirichletAllocation(n_components=t, learning_method='batch',random_state=42)

lda.fit(data)

#We can obtain the resulting topic-term rank after the model is trained:
print(lda.components_)

#Similarly, for each topic, we display the top 10 terms based on their ranks as follows:
terms = count_vector.get_feature_names()


for topic_idx, topic in enumerate(lda.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))


