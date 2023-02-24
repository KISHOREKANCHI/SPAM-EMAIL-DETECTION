import numpy as np
import pandas as pd
# from google.colab import drive
import nltk
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import seaborn as sea
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report




# Load dataset
# drive.mount('/content/gdrive')
# data_set = pd.read_csv('/content/gdrive/MyDrive/spam.csv',encoding='latin-1')
data_set = pd.read_csv('spam.csv',encoding='latin-1')




# Print first 5 rows
print("Top 5 rows of dataset")
print(data_set.head())

# Print random 10 rows
print("Random 10 rows")
print(data_set.sample(10))

# Print total number of rows and columns in dataset
print("Total rows and columns in dataset")
print(data_set.shape)

# Print number of rows and columns in dataset separately
print("Total Rows =", data_set.shape[0])
print("Total Columns =", data_set.shape[1])

# Print column names
print("Columns in dataset")
print(data_set.columns)

#Renaming columns for better understanding
print("Renaming columns")
data_set.rename(columns={'v1': 'Variety', 'v2': 'Data'}, inplace=True)
print(data_set.head())

#Encode the target variable 'Variety'
print("Encoding target variable")
encoder = LabelEncoder()
data_set['Variety'] = encoder.fit_transform(data_set['Variety']) #allotting 0,1
print(data_set.head())

#Check information of dataset
print("Dataset information")
print(data_set.info())

#Drop unnecessary columns
print("Dropping extra columns")
data_set.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print(data_set.head())

#Check for duplicate values
print("Checking for duplicate data")
print("Total Duplicated values =", data_set.duplicated().sum())

#Remove duplicate values
data_set = data_set.drop_duplicates(keep='first')

#Check for null values
print("Checking for null values")
print("Total NULL values =\n\n",data_set.isnull().sum())

#Print size of dataset
print("Size of dataset is:", data_set.size)

#Print random 6 rows after data cleaning
print("After DATA CLEANING")
print("Total Rows:", data_set.shape[0], "\nTotal columns:", data_set.shape[1])
print(data_set.sample(6))

# Count spam(1) and non-spam(0) mails
print("Count of spam(1) and non-spam(0) mails")
print(data_set['Variety'].value_counts())




# Generate a word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white').generate(' '.join(data_set['Data']))

# Plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()




#pie chart
fig, ax = plt.subplots(figsize =(5, 5))
#labels
m=['Non-spam','Spam']
#title for chart
ax.set_title("Customizing pie chart",color="red")
# Creating color parameters
colors = ( "grey", "cyan")
# Creating explode data
explode = (0.2, 0.0)
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }
# Creating plot
wedges, texts, autotexts = ax.pie(data_set['Variety'].value_counts(),
                                  autopct = "%0.2f",
                                  explode = explode,
                                  labels = m,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="red"))
# Adding legend
ax.legend(wedges,m,
          title ="MAILS",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
 
plt.setp(autotexts, size = 8, weight ="bold")

plt.show()




#Distribution plot
sea.histplot(data=data_set, x=data_set['Variety'], hue="Variety", multiple="stack", kde=True)
plt.xlabel('Email Length')
plt.ylabel('Count')
plt.title('Distribution of Email Length for Spam and Non-Spam Emails')
plt.show()




#Heatmap
sea.heatmap(data_set.corr(), annot=True)
plt.title('Correlation between Features in Email Dataset')
plt.show()




# Tokenize words
nltk.download('punkt')
data_set['words'] = data_set['Data'].apply(lambda x:len(nltk.word_tokenize(x)))
print(data_set.sample(8))


# Tokenize sentences
data_set['sentence'] = data_set['Data'].apply(lambda x:len(nltk.sent_tokenize(x)))
print(data_set.sample(8))


# Count number of characters in each text
data_set['chars']= data_set['Data'].apply(len)
print(data_set.sample(8))


# Statistics summary of Spam mails
print("Statistics summary of Spam mails")
print(data_set[data_set['Variety'] == 1][['words', 'sentence', 'chars']].describe())


# Statistics summary of Non-Spam mails
print("Statistics summary of Non-Spam mails")
print(data_set[data_set['Variety'] == 0][['words', 'sentence', 'chars']].describe())


#Removing stop words
nltk.download('stopwords')
stop = stopwords.words('english')
data_set['Data'] = data_set['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_set['Data'].head()


#Removing punctuations and lower casing
data_set['Data'] = data_set['Data'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
data_set['Data'] = data_set['Data'].apply(lambda x: x.lower())


#stemming of words
st = PorterStemmer()
data_set['Data'] = data_set['Data'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
data_set.head()


#Vectorizing the Words
tf_vec = TfidfVectorizer()
features = tf_vec.fit_transform(data_set['Data'])
X = features
y = data_set['Variety']


#Splitting the Dataset into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#Model Training
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)




#prediction
prediction_on_training_data = knn.predict(X_train)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data*100)




#Model Training
clf = MultinomialNB()
clf.fit(X_train, y_train)




#Prediction and Model Evaluation
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy:', accuracy*100)
print('Confusion matrix:\n', conf_matrix)
print('Classification report:\n', class_report)