# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import streamlit as st

# Title of the Streamlit app
st.title("Amazon Product Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing Amazon product data", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Step 1: Handle Missing Values
st.subheader("Data Cleaning and Preprocessing")
critical_columns = ['product_id', 'actual_price', 'discounted_price', 'rating', 'rating_count', 'category']
missing_values = df[critical_columns].isnull().sum()
st.write("Missing values before cleaning:", missing_values)

df_cleaned = df.dropna(subset=critical_columns)

# Remove non-numeric characters and normalize data
df_cleaned['actual_price'] = df_cleaned['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df_cleaned['discounted_price'] = df_cleaned['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df_cleaned['rating'] = pd.to_numeric(df_cleaned['rating'], errors='coerce')
df_cleaned['rating_count'] = df_cleaned['rating_count'].str.replace(',', '').astype(float)

imputer = SimpleImputer(strategy='median')
df_cleaned['rating'] = imputer.fit_transform(df_cleaned[['rating']])

scaler = StandardScaler()
numerical_columns = ['actual_price', 'discounted_price', 'rating', 'rating_count']
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])

# Encode categorical features
df_cleaned['main_category'] = df_cleaned['category'].str.split('|').str[0]
encoder = LabelEncoder()
df_cleaned['main_category_encoded'] = encoder.fit_transform(df_cleaned['main_category'])

st.write("Sample cleaned data:")
st.dataframe(df_cleaned.head())

# Step 2: Data Visualization
st.subheader("Data Visualization")

# Distribution of Ratings
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_cleaned['rating'], bins=20, kde=True, ax=ax)
ax.set_title('Distribution of Ratings')
st.pyplot(fig)

# Clustering using K-Means
features = ['discounted_price', 'actual_price', 'main_category_encoded', 'rating', 'rating_count']
X = df_cleaned[features]
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 11), inertia, marker='o', linestyle='-', color='blue')
ax.set_title('Elbow Method for Optimal Number of Clusters')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
st.pyplot(fig)

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_cleaned['customer_segment'] = kmeans.fit_predict(X_scaled)

# PCA for cluster visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cleaned['PCA1'] = X_pca[:, 0]
df_cleaned['PCA2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='customer_segment', data=df_cleaned, palette='viridis', ax=ax)
ax.set_title('Customer Segments Visualization using PCA')
st.pyplot(fig)

# Association Rule Mining
transactions = df_cleaned.groupby(['product_id', 'product_name'])['main_category'].apply(list).values.tolist()
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules = rules.sort_values(by='lift', ascending=False)

st.write("Top Association Rules:")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Sentiment Analysis
st.subheader("Sentiment Analysis")
df_cleaned['sentiment'] = df_cleaned['review_content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_cleaned['sentiment'], bins=20, color='purple', kde=True, ax=ax)
ax.set_title('Sentiment Distribution')
st.pyplot(fig)

# Word Cloud
review_text = ' '.join(df_cleaned['review_content'].astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(review_text)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud of Review Content')
st.pyplot(fig)

# Customer Segments by Sentiment
df_cleaned['sentiment_label'] = pd.cut(df_cleaned['sentiment'], bins=[-1, -0.01, 0.01, 1], labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=df_cleaned, palette='viridis', ax=ax)
ax.set_title('Customer Segments based on Sentiment')
st.pyplot(fig)

