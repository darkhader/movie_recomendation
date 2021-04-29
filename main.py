# mount your drive
from google.colab import drive
drive.mount('/content/drive')

# read the CSV file
md = pd. read_csv('drive/My Drive/Colab Notebooks/Movie_recommendation/movie_dataset/movies_metadata.csv')
# droping rows by index
md = md.drop([19730, 29503, 35587])

#performing look up opertion on all movies that are present in links_small dataset
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd.shape

smd['tagline'] = smd['tagline'].fillna(' ')
smd['tagline']
# Merging Overview and tittle together
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna(' ')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

# Cosine similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd['title']
# finding indices of every title
indices = pd.Series(smd.index, index=titles)

# function that returns the 30 most similar movies based on the cosine similarity score.
from flask import Flask
app = Flask(__name__)

@app.route("/")
def main():
    title = request.args.get('movie')
    idx = indices[title]
    print("Index",idx)
    similar_scores = list(enumerate(cosine_sim[idx]))

    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:6]
    movie_indices = [i[0] for i in similar_scores]

    output = []
    for item in titles.iloc[movie_indices]:
        output.append(item)
    return json.dumps(output)
if __name__ == "__main__":
    app.run()