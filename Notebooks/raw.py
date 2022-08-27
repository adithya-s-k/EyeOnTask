#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import cufflinks as cf
import numpy as np
import pandas as pd
from langdetect import detect
import matplotlib.pyplot as plt
import string
import re
valid_chars = string.ascii_letters+string.digits+' '
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as pyoff
import plotly.graph_objs as go

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

pyoff.init_notebook_mode()
cf.go_offline()


# In[2]:


bookdata_path = '/Users/ernestng/Desktop/projects/bookLSTM/scrapedata/book_data.csv'
testdata_path = '/Users/ernestng/Desktop/projects/bookLSTM/scrapedata/book_data18.csv'
book = pd.read_csv(bookdata_path)
test = pd.read_csv(testdata_path)
book.head()


# #### get number of genres for each book

# In[3]:


def genre_count(x):
    try:
        return len(x.split('|'))
    except:
        return 0

book['genre_count'] = book['genres'].map(lambda x: genre_count(x))
test['genre_count'] = test['genres'].map(lambda x: genre_count(x))
book.head()


# In[4]:


plot_data = [
    go.Histogram(
        x=book['genre_count']
    )
]
plot_layout = go.Layout(
        title='Genre distribution',
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Number of Genres"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# #### most books have approximately 5-6 genres each and these are user-defined
# 
# #### I want to see the number of unique genres across the whole dataset so that I can see the variety in my dataset.

# In[5]:


#make a genre columns into a list of all genres
def genre_listing(x):
    try:
        lst = [genre for genre in x.split("|")]
        return lst
    except: 
        return []

book['genre_list'] = book['genres'].map(lambda x: genre_listing(x))
test['genre_list'] = test['genres'].map(lambda x: genre_listing(x))


# In[6]:


genre_dict = defaultdict(int)
for idx in book.index:
    g = book.at[idx, 'genre_list']
    if type(g) == list:
        for genre in g:
            genre_dict[genre] += 1
genre_dict


# In[7]:


len(genre_dict)


# we have 866 unique genres across our entire dataset. Now I want to see the top few genres

# In[8]:


genre_pd = pd.DataFrame.from_records(sorted(genre_dict.items(), key=lambda x:x[1], reverse=True), 
                                     columns=['genre', 'count'])
genre_pd[:50].head()


# In[9]:


plot_data = [
    go.Bar(
        x=genre_pd['genre'],
        y=genre_pd['count']
    )
]
plot_layout = go.Layout(
        title='Distribution for all Genres',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Genre"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# it is not practical to look at those genres with very low counts since it holds little to no value to us. I will only want to look at the top unique genres that is representative of the datasetm hence I can pick the top 50 genres to look at

# In[10]:


plot_data = [
    go.Bar(
        x=genre_pd[:50]['genre'],
        y=genre_pd[:50]['count']
    )
]
plot_layout = go.Layout(
        title='Distribution for Top 50 genres',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Genre"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# If we look at the genre_list column, a book is classfied as fiction if 'fiction' is included as at least one of its genres. By observation, if a book has at least fiction in its genre_list, all other genres in the same list will be closely associated to fiction as well. From this I can compare the number of fiction and nonfiction books in my dataset.

# In[11]:


def determine_fiction(x):
    lower_list = [genre.lower() for genre in x]
    if 'fiction' in lower_list:
        return 'fiction'
    elif 'nonfiction' in lower_list:
        return 'nonfiction'
    else:
        return 'others'
book['label'] = book['genre_list'].apply(determine_fiction)
test['label'] = test['genre_list'].apply(determine_fiction)


# since my aim to predict if a book is fiction or nonfiction based on their description, I want to make sure there are no formatting errors in each description. If there happens to be any other language other than english/missing values/non-ascii characters, I must make sure to identify them. 

# In[12]:


def remove_invalid_lang(df):
    '''
    Removes records that have invalid descriptions from the dataframe
    Input: dataframe
    Output: Cleaned up dataframe
    '''
    invalid_desc_idxs=[]
    for i in df.index:
        try:
            a=detect(df.at[i,'book_desc'])
        except:
            invalid_desc_idxs.append(i)
    
    df=df.drop(index=invalid_desc_idxs)
    return df


# In[13]:


book = remove_invalid_lang(book)


# In[14]:


test = remove_invalid_lang(test)


# In[15]:


test['lang']=test['book_desc'].map(lambda desc: detect(desc))


# In[16]:


book['lang']=book['book_desc'].map(lambda desc: detect(desc))
book.head()


# get languages from wikipedia

# In[17]:


#Downloading the list of languages to map the two-letter lang code to the language name
lang_lookup = pd.read_html('https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes')[1]
langpd = lang_lookup[['ISO language name','639-1']]
langpd.columns = ['language','iso']


# In[18]:


langpd


# In[19]:


def desc_lang(x):
    if x in list(langpd['iso']):
        return langpd[langpd['iso'] == x]['language'].values[0]
    else:
        return 'nil'
book['language'] = book['lang'].apply(desc_lang)
book.head(5)


# In[20]:


test['language'] = test['lang'].apply(desc_lang)


# In[21]:


plot_data = [
    go.Histogram(
        x=book['language']
    )
]
plot_layout = go.Layout(
        title='Distribution for languages',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Language"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# We see that the majority of books are in English and since we are going to base our predictions on english descriptions only, I will remove all non-English books. 

# In[22]:


nonen_books = book[book['language']!='English']
plot_data = [
    go.Histogram(
        x=nonen_books['language']
    )
]
plot_layout = go.Layout(
        title='Distribution for non English books',
        yaxis= {'title': "Count"},
        xaxis= {'title': "Language"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[23]:


test = test[test['language']=='English']


# In[24]:


en_books = book[book['language']=='English']


# checkpoint

# In[25]:


en_books.to_csv('/Users/ernestng/Desktop/projects/bookLSTM/scrapedata/checkpoint.csv')


# In[26]:


test.to_csv('/Users/ernestng/Desktop/projects/bookLSTM/scrapedata/testcheckpoint.csv')


# #### we have to clean each record:
# #### 1) some may contain non-ascii characters
# #### 2) no genre specified
# #### 3) no language assigned

# In[27]:


en_books = pd.read_csv('/Users/ernestng/Desktop/projects/bookLSTM/scrapedata/checkpoint.csv')
en_books


# Function to remove non ascii characters and clean text

# In[28]:


def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def cleaner(df):
   
    df = df[df['label'] != 'others']

    df = df[df['language'] != 'nil']

    df['clean_desc'] = df['book_desc'].apply(clean_text)

    return df


# In[29]:


clean_book = cleaner(en_books)


# In[30]:


clean_test = cleaner(test)


# get length of description for each book

# In[31]:


clean_book['desc_len'] = [len(i.split()) for i in clean_book.clean_desc]
clean_book.head(3)


# In[32]:


plot_data = [
    go.Histogram(
        x=clean_book['desc_len']
    )
]
plot_layout = go.Layout(
        title='Distribution of description length',
        yaxis= {'title': "Length"},
        xaxis= {'title': "Descriptions"}
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# I observe that most of descriptions have length below approx 500. We have to make sure that we use descriptions with equal length, this is to allow the creation of fixed shape tensors when we create our X-train matrix and this ensures more stable weights assigned to each word during our model training phase. Hence I aim to do clipping and padding for each description. Firstly I would have to determine the optimal length.

# In[33]:


len_df_bins=clean_book.desc_len.value_counts(bins=100, normalize=True).reset_index().sort_values(by=['index'])
len_df_bins['cumulative']=len_df_bins.desc_len.cumsum()
len_df_bins['index']=len_df_bins['index'].astype('str')
len_df_bins.iplot(kind='bar', x='index', y='cumulative')


# about 83% of the records have word count of lesser than 216 words. Hence I will set my max threshold at 200. We will need a min threshold as well and I will set that to 6 since any description with less than 5 words is unlikely to be descriptive enough to determine the genre. 
# 
# ### Clipping and Padding
# 
# #### for records where the description is less than 200 words, we will pad them with empty values, whereas for records where the description is more than 200 words, we will clip them to include just the first 200. 
# 
# #### I will use the integer zero for padding.
# 
# #### The RNN will read the token sequence left-to-right and will output a single prediction for whether the book is fiction or nonfiction. The memory of these tokens are passed on one by one to the final token, and thus, it is important to pre-pad the sequence instead of post-padding it. This means that the zeros are added BEFORE the token sequence and not after. There are situations where post-padding may be more effective, for example, in bi-directional networks.

# In[34]:


min_desc_length=6
max_desc_length=200

clean_book=clean_book[(clean_book.clean_desc.str.split().apply(len)>min_desc_length)].reset_index(drop=True)


# In[35]:


clean_test=clean_test[(clean_test.clean_desc.str.split().apply(len)>min_desc_length)].reset_index(drop=True)


# In[36]:


clean_book.head()


# #### build vocab and assign integers to words(tokenizing)

# In[37]:


vocabulary=set() #unique list of all words from all description

def add_to_vocab(df, vocabulary):
    for i in df.clean_desc:
        for word in i.split():
            vocabulary.add(word)
    return vocabulary

vocabulary=add_to_vocab(clean_book, vocabulary)

#This dictionary represents the mapping from word to token. Using token+1 to skip 0, since 0 will be used for padding descriptions with less than 200 words
vocab_dict={word: token+1 for token, word in enumerate(list(vocabulary))}

#This dictionary represents the mapping from token to word
token_dict={token+1: word for token, word in enumerate(list(vocabulary))}

assert token_dict[1]==token_dict[vocab_dict[token_dict[1]]]

def tokenizer(desc, vocab_dict, max_desc_length):
    '''
    Function to tokenize descriptions
    Inputs:
    - desc, description
    - vocab_dict, dictionary mapping words to their corresponding tokens
    - max_desc_length, used for pre-padding the descriptions where the no. of words is less than this number
    Returns:
    List of length max_desc_length, pre-padded with zeroes if the desc length was less than max_desc_length
    '''
    a=[vocab_dict[i] if i in vocab_dict else 0 for i in desc.split()]
    b=[0] * max_desc_length
    if len(a)<max_desc_length:
        return np.asarray(b[:max_desc_length-len(a)]+a).squeeze()
    else:
        return np.asarray(a[:max_desc_length]).squeeze()


# In[38]:


len(vocabulary)


# In[39]:


clean_test['desc_tokens']=clean_test['clean_desc'].apply(tokenizer, args=(vocab_dict, max_desc_length))


# In[40]:


clean_book['desc_tokens']=clean_book['clean_desc'].apply(tokenizer, args=(vocab_dict, max_desc_length))
clean_book.head(2)


# ### Training and validation data sets
# #### When the dataset is imbalanced, i.e., the distribution of target variable (fiction/nonfiction) is not uniform, we should make sure that the training-validation split is stratified. This ensures that the distribution of the target variable is preserved in both the training and validation datasets.

# In[41]:


clean_book.label.value_counts()


# We can also try random undersampling to reduce number of fiction samples However, I will use stratified sampling.
# 
# Stratified random samples are used with populations that can be easily broken into different subgroups or subsets, in our case, fiction or nonfiction. I will randomly choose record from each label in proportion to the group's size versus the population. Each record must only belong to one stratum(label) and I am certain each record is mutually exclusive since a book can only either be fiction or nonfiction. Overlapping strata would increase the likelihood that some data are included, thus skewing the sample.
# 
# One advantage over random undersampling is that because it uses specific characteristics, it can provide a more accurate representation of the books based on what's used to divide it into different subsets, also we don't have to remove which any records which might be useful in our model.

# In[42]:


def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs


# In[43]:


_, sample_idxs = stratified_split(clean_book, 'label', 0.1)

train_idxs, val_idxs = stratified_split(clean_book, 'label', val_percent=0.2)
sample_train_idxs, sample_val_idxs = stratified_split(clean_book[clean_book.index.isin(sample_idxs)], 'label', val_percent=0.2)


# function to ensure that my stratified sampling method is working correctly i.e proportion of fiction and nonfiction labels in train and validation set is preserved

# In[44]:


def test_stratified(df, col):
    '''
    Analyzes the ratio of different classes in a categorical variable within a dataframe
    Inputs:
    - dataframe
    - categorical column to be analyzed
    Returns: None
    '''
    classes=list(df[col].unique())
    
    for c in classes:
        print(f'Proportion of records with {c}: {len(df[df[col]==c])*1./len(df):0.2} ({len(df[df[col]==c])} / {len(df)})')
    print("----------------------")


# In[45]:


test_stratified(clean_book, 'label')
test_stratified(clean_book[clean_book.index.isin(train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(val_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_train_idxs)], 'label')
test_stratified(clean_book[clean_book.index.isin(sample_val_idxs)], 'label')


# In[46]:


classes=list(clean_book.label.unique())
classes


# In[47]:


sampling=False

x_train=np.stack(clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['desc_tokens'])
y_train=clean_book[clean_book.index.isin(sample_train_idxs if sampling else train_idxs)]['label'].apply(lambda x:classes.index(x))

x_val=np.stack(clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['desc_tokens'])
y_val=clean_book[clean_book.index.isin(sample_val_idxs if sampling else val_idxs)]['label'].apply(lambda x:classes.index(x))


# In[48]:


x_test=np.stack(clean_test['desc_tokens'])
y_test=clean_test['label'].apply(lambda x:classes.index(x))


# ### Model Building
# I will be using a recurrent neural network with one embedding layer, 2 LSTM layers, a dense layer with sigmoid activation to classify the book description as either fiction or nonfiction.
# 
# The embedding layer helps us reduce the dimensionality of the problem. If we one-hot encode the words in the vocabulary, each word will be represented by a vector the size of the vocabulary itself, which in this case is 85643. Since each sample will be a tensor of size (vocabulary x no. of tokens), i.e., (85643 x 200), the size of the layer will be too big for the LSTM to consume and it will be very resource-intensive and time-consuming for the training process. If I use embedding, my tensor size will only 200 x 200.
# 
# One hot encoding will result in a huge sparse matrix while embedding gives us a dense representation. The higher the embedding length, the more complex representations our model can learn since our embedding layer learns a "representation" of each word that is of a fixed length. 

# #### How many LSTM layers should we use? Speed-Complexity tradeoff
# 
# Usually 1 layer is enough to find trends in simple problems and 2 is sufficient to find reasonably complex features. We can compare accuracy of our model after a fixed number of epochs for a range of choices(number of layers), if we find that the accuracy does not change significantly even after adding more layers, we can select the minimum number of layers. 

# In[49]:


#initialize model and add embedding layer
model = Sequential()
# decide on number of hidden nodes
model.add(Embedding(len(vocabulary)+1, output_dim=200, input_length=max_desc_length))


# #### How many hidden nodes should we add to our LSTM layers?
# Ns : number of samples in training data
# Ni : number of input neurons
# No : number of output neurons
# alpha : scaling factor(indicator of how general you want your model to be, or how much you want to prevent overfitting)
# 
# General formula: Ns / [alpha * (Ni + No)]

# #### Adding a Dropout layer. Accuracy-Overfit prevention tradeoff
# 
# Prevents overfitting by ignoring randomly selected neurons during training
# Reduces sensitivity to specific weights of individual neurons

# #### Adding a Dense layer
# 
# Since we have 1 output label(fiction or nonfiction), we will have 1 output label.

# #### Adding Activation layer
# 
# There are many activation functions to choose from so it depends on our goal.
# 
# In this case, we want the output to be either fiction/nonfiction hence Softmax or Sigmoid functions will be good.
# 
# Sigmoid function basically outputs probabilites, we will usually use sigmoid for multi class classification. 
# 
# Softmax function output values between 0 and 1 such that the summation of all output values equals to 1. Basically you get a probability of each class, (join distribution and a multinomial likelihood) whose sum is bound to be one. 
# 
# Since this is a binary classification problem, I will use softmax.

# #### Choosing the loss function, optimizer and judgement metrics
# 
# Since we are faced with a binary classification problem, binary cross-entropy will work well with softmax because the cross-entropy function cancels out the plateaus at each end of the soft-max function and therefore speeds up the learning process.
# 
# For optimizers, adaptive moment estimation(adam), has been shown to work well in most practical applications and works well with only little changes in the hyperparameters.
# 
# Judging the models’ performance from an overall accuracy point of view will be the option easiest to interpret as well as sufficient in resulting model performance. 

# #### Additional metrics

# In[50]:


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[51]:


parameters = {'vocab': vocabulary,
              'eval_batch_size': 30,
              'batch_size': 200,
              'epochs': 5,
              'dropout': 0.2,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookModel1 = bookLSTM(x_train, y_train, x_val, y_val, parameters)


# I notice that as my epochs goes from 3 to 5, test accuracy increases, but validation accuracy decreases. This means that the model is fitting the training set better, but it is losing the ability to predict on new data, indicating that my model is starting to fit on noise and is beginning to overfit.
# 
# Hence approximately 2 epochs should be enough in this case

# In[52]:


parameters = {'vocab': vocabulary,
              'eval_batch_size': 30,
              'batch_size': 128,
              'epochs': 2,
              'dropout': 0.2,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model2"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookModel2 = bookLSTM(x_train, y_train, x_val, y_val, parameters)


# In[53]:


parameters = {'vocab': vocabulary,
              'eval_batch_size': 40,
              'batch_size': 128,
              'epochs': 3,
              'dropout': 0.3,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model3"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookModel3 = bookLSTM(x_train, y_train, x_val, y_val, parameters)


# In[54]:


parameters = {'vocab': vocabulary,
              'eval_batch_size': 40,
              'batch_size': 128,
              'epochs': 5,
              'dropout': 0.4,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model4"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookModel4 = bookLSTM(x_train, y_train, x_val, y_val, parameters)


# In[55]:


parameters = {'vocab': vocabulary,
              'eval_batch_size': 40,
              'batch_size': 128,
              'epochs': 5,
              'dropout': 0,
              'optimizer': 'Adam',
              'loss': 'binary_crossentropy',
              'activation':'sigmoid'}

def bookLSTM(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.name="Book Model5"
    model.add(Embedding(len(params['vocab'])+1, output_dim=x_train.shape[1], input_length=x_train.shape[1]))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(200))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['loss'],
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, 
          y_train,
          validation_data=(x_val, y_val),
          batch_size=params['batch_size'], 
          epochs=params['epochs'])
    results = model.evaluate(x_test, y_test, batch_size=params['eval_batch_size'])
    return model

BookModel5 = bookLSTM(x_train, y_train, x_val, y_val, parameters)


# Test with a fantasy book description - should return 'fiction'

# In[56]:


fantasy='In this mesmerizing sequel to the New York Times bestselling Girls of Paper and Fire, Lei and Wren have escaped their oppressive lives in the Hidden Palace, but soon learn that freedom comes with a terrible cost. Lei, the naive country girl who became a royal courtesan, is now known as the Moonchosen, the commoner who managed to do what no one else could. But slaying the cruel Demon King wasnt the end of the plan---its just the beginning. Now Lei and her warrior love Wren must travel the kingdom to gain support from the far-flung rebel clans. The journey is made even more treacherous thanks to a heavy bounty on Leis head, as well as insidious doubts that threaten to tear Lei and Wren apart from within.Meanwhile, an evil plot to eliminate the rebel uprising is taking shape, fueled by dark magic and vengeance. Will Lei succeed in her quest to overthrow the monarchy and protect her love for Wren, or will she fall victim to the sinister magic that seeks to dest'


# In[57]:


def reviewBook(model,text):
    labels = ['fiction', 'nonfiction']
    a = clean_text(fantasy)
    a = tokenizer(a, vocab_dict, max_desc_length)
    a = np.reshape(a, (1,max_desc_length))
    output = model.predict(a, batch_size=1)
    score = (output>0.5)*1
    pred = score.item()
    return labels[pred]


# In[80]:


reviewBook(BookModel2,fantasy)


# In[82]:


x_train.shape


# In[ ]:




