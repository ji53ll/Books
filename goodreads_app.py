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

################################## stopwords for word cloud ##########################################
import nltk
nltk.download('stopwords')

################################## overall set up ##########################################
st.set_page_config(
    page_title='Your Goodreads Analysis',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='collapsed'
)
st.set_option('deprecation.showPyplotGlobalUse', False)

################################## color theming ###########################################

color_scale_for_bars = 'Blues'
color_scale_for_wordcloud = 'Blues'
# Define your color for the histogram
histogram_color = '#94D7F2' 

################################## set animation ###########################################
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
#file_url = 'https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json'
file_url = 'https://lottie.host/d444a43b-2c5f-4709-b100-8708bc6bb344/FDGGNBK4tN.json'
lottie_book = load_lottieurl(file_url)
st_lottie(lottie_book, speed=1, height=200, key='initial')

#################################### set banner ############################################
# Set the background color and height of the banner
banner_style = """
    background-color: #94D7F2;  /* You can use any valid color representation */
    color: white;
    text-align: center;
    padding: 10px;
    border-radius: 10px;  /* Optional: adds rounded corners to the banner */
"""

# Use HTML to create the banner
st.write(
    '<div style="{}"><h1>Books in Review</h1></div>'.format(banner_style),
    unsafe_allow_html=True,
)

########################### sub text ##########################################################

st.subheader('Web app by [Jisell Howe](https://www.jisellhowe.com)')
"""
This app analyzes (but does not store) the books you've read using Goodreads. Upload your data to see your own insights.
"""
########################### establish columns ##########################################################

# Assuming half the screen width for each column
left_col, right_col = st.columns(2)

# File uploader in the second half
goodreads_file = right_col.file_uploader('## Please Import Your Goodreads Data')


if goodreads_file is None:
    books_df = pd.read_csv('goodreads_library_export_JH.csv')
    left_col.write("### Analyzing Jisell's Goodreads History")
else:
    books_df = pd.read_csv(goodreads_file)
    left_col.write('### Analyzing your Goodreads History')

########################### ore-processing on titles for word cloud ##########################################################
    
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
wordcloud = WordCloud(width=700, height=1067, background_color='white',colormap=color_scale_for_wordcloud).generate(text)


############################ initial column set up in dataframe ####################################################################

books_df['days_to_finish'] = (pd.to_datetime(
    books_df['Date Read']) - pd.to_datetime(books_df['Date Added'])).dt.days
books_finished_filtered = books_df[(books_df['Exclusive Shelf'] == 'read') & (books_df['days_to_finish'] >= 0)]
u_books = len(books_finished_filtered['Author'].unique())
u_authors = len(books_finished_filtered['Author'].unique())
mode_author = books_finished_filtered['Author'].mode()[0]
right_col.write(f'###### It appears you have finished {u_books} books with a total of {u_authors} unique authors. Your most read author is {mode_author}.')
left_col.write(f'###### Your app results can be found below.')

books_df['Year Finished'] = pd.to_datetime(books_df['Date Read']).dt.year
books_per_year = books_df.groupby('Year Finished')['Book Id'].count().reset_index()
books_per_year.columns = ['Year Finished', 'Count']

########################### histogram - books read - days to finish ##########################################################
fig_days_finished = px.histogram(books_finished_filtered, 
                                 x='days_to_finish',
                                 labels={'days_to_finish':'days'},
                                 color_discrete_sequence=[histogram_color])
fig_days_finished.update_layout(yaxis_title='',showlegend=False)


########################### histogram - books read - number of pages ##########################################################

fig_num_pages = px.histogram(books_finished_filtered,
                             x='Number of Pages',
                             color_discrete_sequence=[histogram_color])
fig_num_pages.update_layout(yaxis_title='',showlegend=False)

########################### gather titles for hover data - published ##########################################################
def aggregate_all_titles_pub(dataframe, title_col):
    return dataframe.groupby('Year Published')[title_col].agg(lambda x: '<br>'.join(x)).reset_index(name='All Titles')


# Aggregate all titles
all_titles_pub = aggregate_all_titles_pub(books_df, 'Title')


#####

books_publication_year = books_df.groupby('Year Published')['Book Id'].count().reset_index()
books_publication_year.columns = ['Year Published','Count']

# Create a new DataFrame for each title for 'Year Published'
df_list_pub = []
for index, row in books_publication_year.iterrows():
    year = row['Year Published']
    titles = all_titles_pub[all_titles_pub['Year Published'] == year]['All Titles'].iloc[0]
    count = len(titles) if isinstance(titles, list) else 1
    df_list_pub.append(pd.DataFrame({'Year Published': [year] * count, 'Title': titles}))

# Concatenate the DataFrames for 'Year Published'
result_df_pub = pd.concat(df_list_pub, ignore_index=True)

# Merge with all_titles_pub using the index
books_publication_year = pd.merge(books_publication_year, all_titles_pub, how='left', left_index=True, right_index=True)

# Drop rows with NaN values in the "All Titles" column for 'Year Published'
books_publication_year = books_publication_year.dropna(subset=['All Titles'])

# Rename columns to avoid conflicts for 'Year Published'
books_publication_year = books_publication_year.rename(columns={'Year Published_x': 'Year Published'})

####

########################### bar chart - published ##########################################################

fig_year_published = px.bar(
                            books_publication_year,
                            x='Year Published',
                            y='Count',
                            color='Count',
                            color_discrete_map=color_scale_for_bars,
                            hover_data=['All Titles']
                            )
fig_year_published.update_xaxes(range=[1980,2024])
fig_year_published.update_layout(yaxis_title='',showlegend=False,coloraxis_showscale=False)
# Remove color bar legend
fig_year_published.update_coloraxes(colorbar=dict(title='', tickvals=[], ticktext=[]))

#####
#####

########################### gather titles for hover data - finished ##########################################################

def aggregate_all_titles_fin(dataframe, title_col):
    return dataframe.groupby(dataframe['Year Finished'])[title_col].agg(lambda x: '<br>'.join(x)).reset_index(name='All Titles')



# Aggregate all titles
all_titles_fin = aggregate_all_titles_fin(books_df, 'Title')



books_per_year = books_df.groupby('Year Finished')['Book Id'].count().reset_index()
books_per_year.columns = ['Year Finished','Count']


# Create a new DataFrame for each title for 'Year Finished'
df_list_fin = []
for index, row in books_per_year.iterrows():
    year = row['Year Finished']
    titles = all_titles_fin[all_titles_fin['Year Finished'] == year]['All Titles'].iloc[0]
    count = len(titles) if isinstance(titles, list) else 1
    df_list_fin.append(pd.DataFrame({'Year Finished': [year] * count, 'Title': titles}))

# Concatenate the DataFrames for 'Year Finished'
result_df_fin = pd.concat(df_list_fin, ignore_index=True)

# Merge with all_titles_fin using the index
books_per_year = pd.merge(books_per_year, all_titles_fin, how='left', left_index=True, right_index=True)

# Drop rows with NaN values in the "All Titles" column for 'Year Finished'
books_per_year = books_per_year.dropna(subset=['All Titles'])

# Rename columns to avoid conflicts for 'Year Finished'
books_per_year = books_per_year.rename(columns={'Year Finished_x': 'Year Finished'})


books_per_year = books_df.groupby('Year Finished')['Book Id'].count().reset_index()
books_per_year.columns = ['Year Finished','Count']

# Create a new DataFrame for each title

df_list = []
for index, row in books_per_year.iterrows():
    year = row['Year Finished']
    titles = all_titles_fin[all_titles_fin['Year Finished'] == year]['All Titles'].iloc[0]
    count = len(titles) if isinstance(titles, list) else 1
    df_list.append(pd.DataFrame({'Year Finished': [year] * count, 'Title': titles}))

# Concatenate the DataFrames
result_df = pd.concat(df_list, ignore_index=True)

# Reset index before merging
books_per_year = books_per_year.reset_index(drop=True)
all_titles = all_titles_fin.reset_index(drop=True)

# Merge with all_titles using the index
books_per_year = pd.merge(books_per_year, all_titles_fin, how='left', left_index=True, right_index=True)

# Drop rows with NaN values in the "All Titles" column
books_per_year = books_per_year.dropna(subset=['All Titles'])

# Rename columns to avoid conflicts
books_per_year = books_per_year.rename(columns={'Year Finished_x': 'Year Finished'})


########################### bar chart - published ##########################################################

fig_year_finished = px.bar(books_per_year, 
                           x='Year Finished', 
                           y='Count',
                           color='Year Finished',
                           hover_data=['All Titles'],
                           color_discrete_map=color_scale_for_bars)


# Explicitly set the x-axis type to 'category'; get rid of y label & legend
fig_year_finished.update_layout(xaxis_type='category', yaxis_title='',showlegend=False,coloraxis_showscale=False)
# Remove color bar legend
fig_year_finished.update_coloraxes(colorbar=dict(title='', tickvals=[], ticktext=[]))


########################### histograms - my rating and average rating ##########################################################

books_rated = books_df[books_df['My Rating']!= 0]
fig_my_rating = px.histogram(books_rated, 
                             x='My Rating',
                             color_discrete_sequence=[histogram_color])
fig_avg_rating = px.histogram(books_rated,
                             x='Average Rating',
                             color_discrete_sequence=[histogram_color])
fig_my_rating.update_layout(yaxis_title='')
fig_avg_rating.update_layout(yaxis_title='',showlegend=False)



avg_difference = np.round(np.mean(books_rated['My Rating'] - books_rated['Average Rating']),2)
if avg_difference >= 0:
    sign = 'higher'
else:
    sign = 'lower'

########################### assembly of word cloud & all charts ##########################################################
    
st.write("---")
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)
row3_col1, row3_col2 = st.columns(2)
with row1_col1:
     st.write(" #### Your Most Common Book Themes")
     st.image(wordcloud.to_image())
     st.write("---")     
with row1_col2:
     st.write('##### The books you have read or were interested in reading were published in these years. This chart is zoomed into the period of 1980-2023. Zoom in and out for other time periods.')
     st.plotly_chart(fig_year_published)
     st.write("---")
     mode_year_finished = int(books_df['Year Finished'].mode()[0])
     st.write(f'##### You have finished the most books in {mode_year_finished}.')
     st.plotly_chart(fig_year_finished)
     st.write("---")
with row2_col1:
     avg_pages = int(books_df['Number of Pages'].mean())
     st.write(f'##### Your books are an average of {avg_pages} pages long.')
     st.plotly_chart(fig_num_pages)
     st.write("---")
with row2_col2:
     mean_days_to_finish = int(books_finished_filtered['days_to_finish'].mean())
     st.write(f'###### It is currently taking you an average of {mean_days_to_finish} days between when the book was added to Goodreads and when you finish the book')
     st.plotly_chart(fig_days_finished)
     st.write("---")
with row3_col1:
     avg_my_rating = round(books_rated['My Rating'].mean(),2)
     st.write(f'##### You rate an average of {avg_my_rating} stars on Goodreads.')
     st.plotly_chart(fig_my_rating)
with row3_col2:
     st.write(f'##### You rate books {sign} than the average Goodreads user by {abs(avg_difference)}!')
     st.plotly_chart(fig_avg_rating)
     
st.write("---")
