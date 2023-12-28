import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_lottie import st_lottie
import requests
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords (if not already downloaded)
import nltk
nltk.download('stopwords')



st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
file_url = 'https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json'
lottie_book = load_lottieurl(file_url)
st_lottie(lottie_book, speed=1, height=200, key='initial')



st.title('Analyzing Your Goodreads Reading Habits')
st.subheader('Web app by [Jisell Howe](https://www.jisellhowe.com)')
"""
This app analyzes (but does not store) the books you've read using Goodreads. Upload your data to see your own insights.
"""

goodreads_file = st.file_uploader('## Please Import Your Goodreads Data')
if goodreads_file is None:
    books_df = pd.read_csv('goodreads_library_export_JH.csv')
    st.write("# Analyzing Jisell's Goodreads History")
else:
    books_df = pd.read_csv(goodreads_file)
    st.write('# Analyzing your Goodreads History')

#### pre-processing on titles
    
# Combine NLTK and WordCloud stopwords
stop_words = set(stopwords.words('english') + list(STOPWORDS))

# Function to remove stop words and other preprocessing
def preprocess_title(title):
    # Convert to lowercase and remove punctuation (add more cleaning steps if needed)
    title = title.lower().replace('.', '').replace(',', '')

    # Remove stop words
    title = ' '.join([word for word in title.split() if word not in stop_words])

    return title

# Apply preprocessing to the "titles" column
books_df['cleaned_titles'] = books_df['Title'].apply(preprocess_title)


# Combine cleaned titles into a single string
text = ' '.join(books_df['cleaned_titles'])

# Generate word cloud
wordcloud = WordCloud(width=700, height=1000, background_color='white').generate(text)



#####

books_df['Year Finished'] = pd.to_datetime(books_df['Date Read']).dt.year
books_per_year = books_df.groupby('Year Finished')['Book Id'].count().reset_index()
books_per_year.columns = ['Year Finished', 'Count']

fig_year_finished = px.bar(books_per_year, x='Year Finished', y='Count')


# Explicitly set the x-axis type to 'category' and get rid of y label
fig_year_finished.update_layout(xaxis_type='category', yaxis_title='')




#####

books_df['days_to_finish'] = (pd.to_datetime(
    books_df['Date Read']) - pd.to_datetime(books_df['Date Added'])).dt.days
books_finished_filtered = books_df[(books_df['Exclusive Shelf'] == 'read') & (books_df['days_to_finish'] >= 0)]
u_books = len(books_finished_filtered['Author'].unique())
u_authors = len(books_finished_filtered['Author'].unique())
mode_author = books_finished_filtered['Author'].mode()[0]
st.write(f'#### It appears you have finished {u_books} books with a total of {u_authors} unique authors. Your most read author is {mode_author}.')
st.write(f'#### Your app results can be found below.')
row1_col1, row1_col2 = st.columns(2)

fig_days_finished = px.histogram(books_finished_filtered, x='days_to_finish',
                                 labels={'days_to_finish':'days'})
fig_days_finished.update_layout(yaxis_title='')


#####

fig_num_pages = px.histogram(books_df,x='Number of Pages')
fig_num_pages.update_layout(yaxis_title='')


def aggregate_all_titles(dataframe, title_col):
    return dataframe.groupby('Year Published')[title_col].agg(lambda x: ', '.join(x)).reset_index(name='All Titles')


# Aggregate all titles
all_titles = aggregate_all_titles(books_df, 'Title')

#####

books_publication_year = books_df.groupby('Original Publication Year')['Book Id'].count().reset_index()
books_publication_year.columns = ['Year Published','Count']

# Reset index before merging
books_publication_year = books_publication_year.reset_index(drop=True)
all_titles = all_titles.reset_index(drop=True)

# Merge with all_titles using index-based merge
books_publication_year = pd.merge(books_publication_year, all_titles, how='left', left_on=books_publication_year.index, right_on=all_titles.index)

# Drop rows with NaN values in the "Title" column
books_publication_year = books_publication_year.dropna(subset=['All Titles'])

# Rename columns to avoid conflicts
books_publication_year = books_publication_year.rename(columns={'Year Published_x': 'Year Published'})

fig_year_published = px.bar(
                            books_publication_year,
                            x='Year Published',
                            y='Count',
                            hover_data=['All Titles']
                            )
fig_year_published.update_xaxes(range=[1980,2023])
fig_year_published.update_layout(yaxis_title='')




####

books_rated = books_df[books_df['My Rating']!= 0]
fig_my_rating = px.histogram(books_rated, x='My Rating')
fig_avg_rating = px.histogram(books_rated,x='Average Rating')
fig_my_rating.update_layout(yaxis_title='')
fig_avg_rating.update_layout(yaxis_title='')



avg_difference = np.round(np.mean(books_rated['My Rating'] - books_rated['Average Rating']),2)
if avg_difference >= 0:
    sign = 'higher'
else:
    sign = 'lower'

####

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)
row3_col1, row3_col2 = st.columns(2)
with row1_col1:
     st.write(" ### Your Most Common Book Themes")
     st.image(wordcloud.to_image())     
with row1_col2:
     mode_year_finished = int(books_df['Year Finished'].mode()[0])
     st.write(f'#### You have finished the most books in {mode_year_finished}.')
     st.plotly_chart(fig_year_finished)
     mean_days_to_finish = int(books_finished_filtered['days_to_finish'].mean())
     st.write(f'#### It took you an average of {mean_days_to_finish} days between when the book was added to Goodreads and when you finished the book. This is not a perfect metric, as you may have added this book to a "want to read" list.')
     st.plotly_chart(fig_days_finished)
with row2_col1:
     avg_pages = int(books_df['Number of Pages'].mean())
     st.write(f'#### Your books are an average of {avg_pages} pages long.')
     st.plotly_chart(fig_num_pages)
with row2_col2:
     st.write('#### The books you have read were published in these years. This chart is zoomed into the period of 1980-2023. Zoom in and out for other time periods.')
     st.plotly_chart(fig_year_published)
with row3_col1:
     avg_my_rating = round(books_rated['My Rating'].mean(),2)
     st.write(f'#### You rate an average of {avg_my_rating} stars on Goodreads.')
     st.plotly_chart(fig_my_rating)
with row3_col2:
     st.write(f'#### You rate books {sign} than the average Goodreads user by {abs(avg_difference)}!')
     st.plotly_chart(fig_avg_rating)
     

