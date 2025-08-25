import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv('feedback.csv')

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Feedback'].apply(get_sentiment)

# Count of Sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Plotting
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green','red','gray'])
plt.title('Customer Feedback Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedbacks')
plt.show()

# Key Pain Points
negative_feedback = df[df['Sentiment'] == 'Negative']
print("Negative Feedbacks:\n", negative_feedback)
