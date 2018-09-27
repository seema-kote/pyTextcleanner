
"""
Created on Sat Sep 15 03:31:51 2018
@author: Seema Kote
@Email: koteseema94@gmail.com
"""

import re
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextCleaner:
    def __init__(self):
        self.cachedStopWords = stopwords.words('english')
        
    def remove_html_tags(self, tweet):
        """
        Remove HTML tags from tweet
        Input: tweet text
        Output: tweet with removed HTML tags
        """
        return BeautifulSoup(tweet, "lxml").text
    
    def remove_stopwords(self, tweet):
	"""
        Remove stopwords from tweet
        Input: tweet text
        Output: tweet with removed stopwords
        """
        return " ".join([word for word in tweet.split(" ") if word not in self.cachedStopWords])
    
    def filter_pattern(self, pattern, tweet, replacer = "''"):
	"""
        Filter pattern from tweet
        Input: pattern, tweet text, replace pattern
        Output: tweet with filter pattern
        """
        pattern = re.compile(pattern)
        return pattern.sub(replacer, tweet)
        
    def remove_puctuations(self, tweet):
	"""
        Remove punctuations from tweet
        Input: tweet text
        Output: tweet with remove punctuations
        """
        tokenizer = RegexpTokenizer(r'\w+')
        return " ".join(tokenizer.tokenize(tweet))
    
    def lemmetize(self, tweet):
	"""
        Lemmetize words from tweet
        Input: tweet text
        Output: tweet with lemmetizing words
        """
        lmtzr = WordNetLemmatizer()
        return " ".join([lmtzr.lemmatize(word) for word in tweet.split()])
    
    def clean_tweet(self,train_dataset):
	"""
        Clean tweet from dataset
        Input: dataset
        Output: tweets with  cleaned data
        """
        cleaned_tweets = []
        for index,row in train_dataset.iterrows():
            tweet = row['text']
            tweet = self.remove_html_tags(tweet)
            tweet = self.filter_pattern(r'\.{2,}',tweet)
            tweet = self.remove_stopwords(tweet.lower())
            tweet = self.remove_puctuations(tweet)
            tweet = self.filter_pattern(r'(@[A-Za-z0-9]+)', tweet)
            tweet = self.filter_pattern(r'(#[A-Za-z0-9]+)', tweet)
            tweet = self.filter_pattern(r'^https?:\/\/.*[\r\n]*',tweet)
            tweet = self.filter_pattern(r'(.)\1+', tweet, r'\1')
            tweet = self.lemmetize(tweet)
            cleaned_tweets.append(tweet.strip())
        return cleaned_tweets
    
    def calculate_tf_idf(self, cleaned_tweets):
	"""
        calculate Term frequency and inverse document frequency
        Input: dataset
        Output: tweets with calulating tf-idf
        """
        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit_transform(cleaned_tweets)
        return vectorizer
    
    def get_sentiment(self,tweet):
	"""
        obtain all sentiments to dataset
        Input: dataset
        Output: tweets with sentiment score
        """
        score = TextBlob(tweet).sentiment
        if score.polarity == 0.0:
            return 'Netural',score.polarity
        if score.polarity < 0:
            return 'Negative',score.polarity
        else:
            return 'Positive',score.polarity
            
        
        
    
