# Import necessary modules from the Selenium library for web automation
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

# Define your Twitter username and password
my_user = 'YOUR USERNAME'
my_password = 'YOUR PASSWORD'

# Define the search term to look for on Twitter
search_item = 'Liz Truss'

# Initialize the Chrome driver (make sure you have the ChromeDriver installed)
driver = webdriver.Chrome()

# Open the Twitter login page
driver.get("https://twitter.com/i/flow/login")

# Wait until the username field is present and visible, then enter the username
user_id = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//input[@name='text']"))
)
user_id.send_keys(my_user)  # Enter the username
user_id.send_keys(Keys.ENTER)  # Press Enter

# Wait until the password field is present and visible, then enter the password
password_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
)
password_field.send_keys(my_password)  # Enter the password
password_field.send_keys(Keys.ENTER)  # Press Enter

# Wait until the search box is present and visible, then search for the specified item
search_box = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//input[@data-testid='SearchBox_Search_Input']"))
)
search_box.send_keys(search_item)  # Enter the search term
search_box.send_keys(Keys.ENTER)  # Press Enter

# Collect tweets mentioning the search item
all_tweets = set()  # Use a set to avoid duplicate tweets

# Wait until tweets are present and visible
tweets = WebDriverWait(driver, 15).until(
    EC.presence_of_all_elements_located((By.XPATH, '//div[@data-testid="tweetText"]'))
)

# Keep scrolling and collecting tweets until we have more than 50 unique tweets
while True:
    for tweet in tweets:
        all_tweets.add(tweet.text)  # Add the tweet text to the set
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')  # Scroll down
    sleep(3)  # Wait for 3 seconds
    tweets = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH, '//div[@data-testid="tweetText"]')))
    if len(all_tweets) > 50:  # Stop if more than 50 tweets are collected
        break
    sleep(2)  # Wait for 2 seconds

# Convert the set of tweets to a list
all_tweets = list(all_tweets)
# print(all_tweets[0])

# cleaning tweets
# Import necessary libraries for data processing and analysis
import pandas as pd
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set pandas option to display all text in columns
pd.options.display.max_colwidth = 1000

# Get stop words in English
stp_words = stopwords.words('english')
# print('....stop words........',stp_words)

# Create a DataFrame from the collected tweets
df = pd.DataFrame(all_tweets, columns=['tweets'])
# print('......dataframe...........\n',df.head())

one_tweet = df.iloc[4]['tweets']
# print('..........one tweet........\n',one_tweet)
# print('.......clean tweet split........', cleanTweet.split())

from wordcloud import WordCloud  # WordCloud: Provides a visual summary of the most common words in the tweets.
from textblob import TextBlob  # TextBlob: Performs sentiment analysis to understand the emotional tone of the tweets.

# Define a function for cleaning tweets
def TweetCleaning(tweet):
    cleanTweet = re.sub(r"@[a-zA-Z0-9]+","",tweet)  # Remove mentions
    # print('.........cleanTweet1.........', cleanTweet)
    cleanTweet = re.sub(r"#[a-zA-Z0-9\s]+","",cleanTweet)  # Remove hashtags
    # print('.........cleanTweet2.........', cleanTweet)
    cleanTweet = ' '.join(word for word in cleanTweet.split() if word not in stp_words)   # Remove stop words
    # print('........,cleanTweet3............',cleanTweet)
    return cleanTweet

# Define a function to calculate the polarity of a tweet
def calPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity  # Calculate polarity of the tweet

# Define a function to calculate the subjectivity of a tweet
def calSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity  # Calculate subjectivity of the tweet

# Define a function to categorize the sentiment based on polarity
def segmentation(tweet):
    if tweet > 0:
        return 'positive'
    if tweet == 0:
        return 'neutral'
    else:
        return 'negative'

# Clean the tweets and calculate their polarity and subjectivity
df['cleanedTweets'] = df['tweets'].apply(TweetCleaning)  # Apply TweetCleaning function to each tweet
df['tPolarity'] = df['cleanedTweets'].apply(calPolarity)  # Apply calPolarity function to each cleaned tweet
df['tSubjectivity'] = df['cleanedTweets'].apply(calSubjectivity)  # Apply calSubjectivity function to each cleaned tweet
df['segmentation'] = df['tPolarity'].apply(segmentation)  # Apply segmentation function to each polarity

# Display the shape and first few rows of the DataFrame
print(df.shape)
print(df.head())

# Analysis & Visualization

# Create a pivot table to count tweets by their sentiment
df.pivot_table(index=['segmentation'], aggfunc={'segmentation': 'count'})

# Get the top 3 most positive tweets
df.sort_values(by=['tPolarity'], ascending=False).head(3)

# Get the top 3 most negative tweets
df.sort_values(by=['tPolarity'], ascending=True).head(3)

# Get the top 3 most neutral tweets
df[df.tPolarity == 0].head(3)

import matplotlib.pyplot as plt

# Consolidate all cleaned tweets into a single string for word cloud
consolidated = ' '.join(word for word in df['cleanedTweets'])
print('...........consolidate.....', consolidated)

# Generate and display a word cloud
wordCloud = WordCloud(width=400, height=200, random_state=20, max_font_size=119).generate(consolidated)
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Print the count of tweets by sentiment
print(df.groupby('segmentation').count())

import seaborn as sns

# Scatter plot of tweet polarity vs subjectivity
plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')
sns.scatterplot(data=df, x='tPolarity', y='tSubjectivity', s=100, hue='segmentation')
plt.show()

# Count plot of the number of tweets by sentiment
sns.countplot(data=df, x='segmentation')
plt.show()

# Calculate percentages of positive, negative, and neutral tweets
positive = round(len(df[df.segmentation == 'positive']) / len(df) * 100, 1)
negative = round(len(df[df.segmentation == 'negative']) / len(df) * 100, 1)
neutral = round(len(df[df.segmentation == 'neutral']) / len(df) * 100, 1)

# Print percentages
responses = [positive, negative, neutral]
print(responses)

# Create a DataFrame with sentiment predictions and their percentages
response = {'resp': ['mayWin', 'mayLoose', 'notSure'], 'pct': [positive, negative, neutral]}
print(pd.DataFrame(response))
