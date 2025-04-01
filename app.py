from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess dataset
courses = pd.read_csv('Coursera.csv')
courses['Course Name'] = courses['Course Name'].str.replace(' ', ',')
courses['Course Name'] = courses['Course Name'].str.replace(',,', ',')
courses['Course Name'] = courses['Course Name'].str.replace(':', '')
courses['Course Description'] = courses['Course Description'].str.replace(' ', ',')
courses['Course Description'] = courses['Course Description'].str.replace(',,', ',')
courses['Course Description'] = courses['Course Description'].str.replace('_', '')
courses['Course Description'] = courses['Course Description'].str.replace(':', '')
courses['Course Description'] = courses['Course Description'].str.replace('(', '')
courses['Course Description'] = courses['Course Description'].str.replace(')', '')
courses['Skills'] = courses['Skills'].str.replace('(', '')
courses['Skills'] = courses['Skills'].str.replace(')', '')
courses['tags'] = courses['Course Name'] + courses['Difficulty Level'] + courses['Course Description'] + courses['Skills']
courses['tags'] = courses['tags'].str.replace(',', ' ')
courses['tags'] = courses['tags'].apply(lambda x: x.lower())
new_df = courses[['Course Name', 'tags']]
new_df.rename(columns={'Course Name': 'course_name'}, inplace=True)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors_cntv = cv.fit_transform(new_df['tags']).toarray()
tfidf = TfidfVectorizer(stop_words='english')
vectors_tf = tfidf.fit_transform(new_df['tags']).toarray()

# Similarity calculation
similarity_cntv = cosine_similarity(vectors_cntv)
similarity_tf = cosine_similarity(vectors_tf)

def recommend(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity_cntv[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    recommendations = [new_df.iloc[i[0]].course_name for i in course_list]
    return recommendations

def recommend_tf(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity_tf[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    recommendations = [new_df.iloc[i[0]].course_name for i in course_list]
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    course_name = request.form['course_name']
    recommendations = recommend(course_name)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
