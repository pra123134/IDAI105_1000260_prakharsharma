# Import necessary libraries
!pip install -r requirements.txt
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
from sklearn.impute import SimpleImputer # Import SimpleImputer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import streamlit as st

# Title of the Streamlit app
st.title("Amazon Product Data Analysis")

# Load the dataset (using Streamlit's file uploader)
uploaded_file = st.file_uploader("amazon.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) 
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()  # Stop execution if no file is uploaded


# Load the dataset
#file_path = 'amazon.csv'  # Replace with your file path
#df = pd.read_csv(file_path)


# Step 1: Handle Missing Values
# Check for missing values in critical columns
critical_columns = ['product_id', 'actual_price', 'discounted_price', 'rating', 'rating_count', 'category']
print("Missing values before cleaning:\n", df[critical_columns].isnull().sum())


# Drop rows with missing values in critical columns
df_cleaned = df.dropna(subset=critical_columns)


# Step 2: Normalize Numerical Data- # Data Cleaning and Preprocessing
# Remove non-numeric characters and convert to float
df_cleaned['actual_price'] = df_cleaned['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df_cleaned['discounted_price'] = df_cleaned['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df_cleaned['rating'] = pd.to_numeric(df_cleaned['rating'], errors='coerce')
df_cleaned['rating_count'] = df_cleaned['rating_count'].str.replace(',', '').astype(float)


# Impute missing values in 'rating' column with the median
imputer = SimpleImputer(strategy='median')
df_cleaned['rating'] = imputer.fit_transform(df_cleaned[['rating']])


# Normalize numerical columns
scaler = StandardScaler()
numerical_columns = ['actual_price', 'discounted_price', 'rating', 'rating_count']
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])


# Step 3: Encode Categorical Features
# Split categories and take the first as main category
df_cleaned['main_category'] = df_cleaned['category'].str.split('|').str[0]


# Encode the 'main_category' column
encoder = LabelEncoder()
df_cleaned['main_category_encoded'] = encoder.fit_transform(df_cleaned['main_category'])


# Display the cleaned and normalized dataframe
print("Cleaned Data Sample:\n", df_cleaned.head())


# Data Visualization
fig, ax = plt.subplots(figsize=(10, 6))  # Create a Matplotlib figure and axes
sns.histplot(df_cleaned['rating'], bins=20, kde=True, ax=ax)  # Plot on the axes
ax.set_title('Distribution of Ratings')
st.pyplot(fig)  # Display the figure using st.pyplot


# Feature Selection for Clustering
features = ['discounted_price', 'actual_price', 'main_category_encoded', 'rating', 'rating_count']
X = df_cleaned[features]


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# Elbow Method for Optimal Number of Clusters
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cluster_range, inertia, marker='o', linestyle='-', color='blue')
ax.set_title('Elbow Method for Optimal Number of Clusters')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.grid(True)
st.pyplot(fig)


# Apply K-Means Clustering
optimal_clusters = 4  # Select based on the Elbow Method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_cleaned['customer_segment'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = df_cleaned.groupby('customer_segment')[features].mean()
print(cluster_summary)

# Visualize clusters
# Customer Segments (Scatter Plot)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='discounted_price', y='rating', hue='customer_segment', size='rating_count', data=df_cleaned, ax=ax)
ax.set_title('Customer Segments')
st.pyplot(fig)

# Visualize Customer Segments using PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cleaned['PCA1'] = X_pca[:, 0]
df_cleaned['PCA2'] = X_pca[:, 1]


fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='customer_segment', data=df_cleaned, palette='viridis', ax=ax)
ax.set_title('Customer Segments Visualization using PCA')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend(title='Segment')
ax.grid(True)
st.pyplot(fig)


# Analyze Customer Segments
segment_analysis = df_cleaned.groupby('customer_segment')[features].mean()
print("Customer Segment Analysis:\n", segment_analysis)


# Visualize Customer Segment Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='customer_segment', data=df_cleaned, palette='viridis', ax=ax)
ax.set_title('Customer Segment Distribution')
ax.set_xlabel('Customer Segment')
ax.set_ylabel('Count')
st.pyplot(fig)


# Prepare Data for Association Rule Mining
# Group products by transactions
transactions = df_cleaned.groupby(['product_id', 'product_name'])['main_category'].apply(list).values.tolist()


# One-hot encode the transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)


# Apply Apriori Algorithm to find frequent itemsets
min_support = 0.01  # Adjust this value based on your dataset
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)


# Generate Association Rules
min_confidence = 0.2  # Adjust this value as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)


# Sort rules by lift to find the most interesting ones
rules = rules.sort_values(by='lift', ascending=False)


# Display the top rules
print("Top Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))


# Visualize the Support, Confidence, and Lift of Top Rules
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
ax.set_title('Association Rules: Support vs Confidence')
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
fig.colorbar(scatter, label='Lift')  # Add colorbar
ax.grid(True)
st.pyplot(fig)

# Decision Tree Classification Example
X = df_cleaned[numerical_columns]
y = df_cleaned['main_category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred))


# Plot Decision Tree
fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figsize as needed
plot_tree(clf, feature_names=numerical_columns, class_names=encoder.classes_, filled=True, ax=ax)
st.pyplot(fig)

# Statistical Analysis
cat1 = df_cleaned[df_cleaned['main_category_encoded'] == 0]['rating']
cat2 = df_cleaned[df_cleaned['main_category_encoded'] == 1]['rating']
t_stat, p_val = ttest_ind(cat1, cat2, nan_policy='omit')
print(f"T-test: t-stat={t_stat}, p-value={p_val}")


# Calculate Discount Percentage
df_cleaned['discount_percentage'] = ((df_cleaned['actual_price'] - df_cleaned['discounted_price']) / 
                                     df_cleaned['actual_price']) * 100


# Step 1: Histograms and Box Plots for Prices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df_cleaned['actual_price'], bins=30, kde=True, color='blue', ax=axes[0])
axes[0].set_title('Distribution of Actual Prices')
axes[0].set_xlabel('Actual Price')
sns.histplot(df_cleaned['discounted_price'], bins=30, kde=True, color='green', ax=axes[1])
axes[1].set_title('Distribution of Discounted Prices')
axes[1].set_xlabel('Discounted Price')
plt.tight_layout()
st.pyplot(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(y=df_cleaned['actual_price'], color='blue', ax=axes[0])
axes[0].set_title('Box Plot of Actual Prices')
sns.boxplot(y=df_cleaned['discounted_price'], color='green', ax=axes[1])
axes[1].set_title('Box Plot of Discounted Prices')
plt.tight_layout()
st.pyplot(fig)



# Step 2: Scatter Plots for Price Relationships
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='actual_price', y='discounted_price', data=df_cleaned, alpha=0.6, ax=ax)
ax.set_title('Actual Price vs Discounted Price')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Discounted Price')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='actual_price', y='discount_percentage', data=df_cleaned, alpha=0.6, color='red', ax=ax)
ax.set_title('Actual Price vs Discount Percentage')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Discount Percentage')
st.pyplot(fig)


# Step 3: Bar Charts for Rating Distribution and Popular Products
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='rating', data=df_cleaned, palette='viridis', ax=ax)
ax.set_title('Product Rating Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_cleaned['rating_count'], bins=30, kde=True, color='purple', ax=ax)
ax.set_title('Distribution of Rating Count')
ax.set_xlabel('Rating Count')
st.pyplot(fig)


# Step 4: Bar Charts/Pie Charts for Category and Product Popularity
category_counts = df_cleaned['main_category'].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
category_counts.plot(kind='bar', color='orange', ax=ax)
ax.set_title('Product Category Distribution')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 8))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), ax=ax)
ax.set_title('Product Category Distribution (Pie Chart)')
ax.set_ylabel('')  # Remove ylabel
st.pyplot(fig)


# Step 5: Correlation Heatmap
correlation_matrix = df_cleaned[['actual_price', 'discounted_price', 'rating', 'rating_count', 'discount_percentage']].corr()


fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)


# Drop rows with missing values in user-related columns
user_columns = ['user_id', 'user_name', 'review_title', 'review_content', 'rating_count']
df_cleaned = df.dropna(subset=user_columns)

# Drop rows with NaN rating_count
df_cleaned = df_cleaned.dropna(subset=['rating_count'])  

# Basic User Behavior Analysis
# Top Users by Review Count
top_users = df_cleaned['user_name'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top_users.plot(kind='bar', color='blue', ax=ax)
ax.set_title('Top 10 Users by Review Count')
ax.set_xlabel('User Name')
ax.set_ylabel('Number of Reviews')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
ax.grid(True)
st.pyplot(fig)


# Rating Distribution Analysis
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_cleaned['rating_count'], bins=20, color='green', kde=True, ax=ax)
ax.set_title('Distribution of Rating Count')
ax.set_xlabel('Rating Count')
ax.set_ylabel('Frequency')
ax.grid(True)
st.pyplot(fig)


# Sentiment Analysis on Review Content
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


df_cleaned['sentiment'] = df_cleaned['review_content'].apply(get_sentiment)


# Plot Sentiment Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_cleaned['sentiment'], bins=20, color='purple', kde=True, ax=ax)
ax.set_title('Sentiment Distribution of Reviews')
ax.set_xlabel('Sentiment Polarity')
ax.set_ylabel('Frequency')
ax.grid(True)
st.pyplot(fig)


# Word Cloud of Review Content
review_text = ' '.join(df_cleaned['review_content'].astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(review_text)


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Review Content')
plt.axis('off')
plt.show()


# Topic Modeling using Latent Dirichlet Allocation (LDA)
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df_cleaned['review_content'].astype(str))
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)


# Display the Top Words for each Topic
num_words = 10
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i+1}:")
    print([words[j] for j in topic.argsort()[-num_words:][::-1]])

# Word Cloud of Review Content
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_title('Word Cloud of Review Content')
ax.axis('off')
st.pyplot(fig)



# Customer Segmentation based on Sentiment and Rating Count
df_cleaned['sentiment_label'] = pd.cut(df_cleaned['sentiment'], bins=[-1, -0.01, 0.01, 1], labels=['Negative', 'Neutral', 'Positive'])


fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=df_cleaned, palette='viridis',dodge=False, ax=ax)
ax.set_title('Customer Segments based on Sentiment')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Reviews')
ax.grid(True)
st.pyplot(fig)

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=df_cleaned, palette='viridis')
plt.title('Customer Segments based on Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.show()

# Cross-tabulation of Rating Count and Sentiment
rating_sentiment_crosstab = pd.crosstab(df_cleaned['rating_count'], df_cleaned['sentiment_label'])
print("\nRating Count vs Sentiment Analysis:\n", rating_sentiment_crosstab)
