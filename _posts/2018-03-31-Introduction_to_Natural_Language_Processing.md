
When you hear your native language, you intuitively know the meaning of what you heard. However, many people who've tried to learn a second or third language find the process to be much more painful. They have to break the language down into components like tenses in order to understand it better. Many have to take years of language lessons to get to the point where they can have a conversation.

Learning a language is difficult because language has many complex rules. If we want computers to be able to understand language, we either need to explicitly teach computers the rules, or enable the computers to intuit the rules themselves. The former is a lot like learning a second language, and the latter is a lot like learning your native language.

Broadly speakingly, natural language processing is the study of enabling computers to understand human languages. This field may involve teaching computers to automatically score essays, infer grammatical rules, or determine the emotions associated with text.

In this mission, we'll learn some of the basic building blocks of natural langage processing. When we feed a computer written text, it has no idea what that text means. In order for a computer to begin making inferences from it, we'll need to convert the text to a numerical representation. This process will enable the computer to intuit grammatical rules, which is more akin to learning a first language.

We'll explore how to get from written text to a numerical representation, and how we can use that representation to make predictions.

[Hacker News](http://news.ycombinator.com/) is a community where users can submit articles, and other users can upvote those articles. The articles with the most upvotes make it to the front page, where they're more visible to the community.

Our data set consists of submissions users made to Hacker News from 2006 to 2015. Developer Arnaud Drizard used the Hacker News API to scrape the data, which you can find [in one of his GitHub repositories](https://github.com/arnauddri/hn). We've sampled 3000 rows from the data randomly, and removed all of the extraneous columns. Our data only has four columns:

- submission_time - When the article was submitted
- upvotes - The number of upvotes the article received
- url - The base URL of the article
- headline - The article's headline

In this mission, we'll be predicting the number of upvotes the articles received, based on their headlines. Because upvotes are an indicator of popularity, we'll discover which types of articles tend to be the most popular.


```python
import pandas as pd
```


```python
submissions = pd.read_csv("sel_hn_stories.csv")
```


```python
submissions.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2014-06-24T05:50:40.000Z</th>
      <th>1</th>
      <th>flux7.com</th>
      <th>8 Ways to Use Docker in the Real World</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-17T16:57:59Z</td>
      <td>1</td>
      <td>blog.jonasbandi.net</td>
      <td>Software: Sadly we did adopt from the construc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-02-04T02:36:30Z</td>
      <td>1</td>
      <td>blogs.wsj.com</td>
      <td>Google’s Stock Split Means More Control for L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-26T07:11:29Z</td>
      <td>1</td>
      <td>threatpost.com</td>
      <td>SSL DOS attack tool released exploiting negoti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-03T15:43:44Z</td>
      <td>67</td>
      <td>algorithm.com.au</td>
      <td>Immutability and Blocks Lambdas and Closures</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-13T16:49:20Z</td>
      <td>1</td>
      <td>winmacsofts.com</td>
      <td>Comment optimiser la vitesse de Wordpress?</td>
    </tr>
  </tbody>
</table>
</div>




```python
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>submission_time</th>
      <th>upvotes</th>
      <th>url</th>
      <th>headline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-17T16:57:59Z</td>
      <td>1</td>
      <td>blog.jonasbandi.net</td>
      <td>Software: Sadly we did adopt from the construc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-02-04T02:36:30Z</td>
      <td>1</td>
      <td>blogs.wsj.com</td>
      <td>Google’s Stock Split Means More Control for L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-26T07:11:29Z</td>
      <td>1</td>
      <td>threatpost.com</td>
      <td>SSL DOS attack tool released exploiting negoti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-03T15:43:44Z</td>
      <td>67</td>
      <td>algorithm.com.au</td>
      <td>Immutability and Blocks Lambdas and Closures</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-13T16:49:20Z</td>
      <td>1</td>
      <td>winmacsofts.com</td>
      <td>Comment optimiser la vitesse de Wordpress?</td>
    </tr>
  </tbody>
</table>
</div>




```python
submissions = submissions.dropna()
submissions.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>submission_time</th>
      <th>upvotes</th>
      <th>url</th>
      <th>headline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-17T16:57:59Z</td>
      <td>1</td>
      <td>blog.jonasbandi.net</td>
      <td>Software: Sadly we did adopt from the construc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-02-04T02:36:30Z</td>
      <td>1</td>
      <td>blogs.wsj.com</td>
      <td>Google’s Stock Split Means More Control for L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-26T07:11:29Z</td>
      <td>1</td>
      <td>threatpost.com</td>
      <td>SSL DOS attack tool released exploiting negoti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-03T15:43:44Z</td>
      <td>67</td>
      <td>algorithm.com.au</td>
      <td>Immutability and Blocks Lambdas and Closures</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-13T16:49:20Z</td>
      <td>1</td>
      <td>winmacsofts.com</td>
      <td>Comment optimiser la vitesse de Wordpress?</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython import display
```

Our goal is to train a linear regression algorithm that predicts the number of upvotes a headline would receive. To do this, we'll need to convert each headline to a numerical representation.
While there are several ways to accomplish this, we'll use a bag of words model. A bag of words model represents each piece of text as a numerical vector. 
We'll examine each step in the bag of words process in this mission. For now, here's a high-level diagram showing how two sentences, I rode my horse to Berlin. and You rode my horse to Berlin in the winter., convert to a bag of words:
     
![tokenizing-the-headlines.svg](/images/posts/2018_3_31/tokenizing-the-headlines.svg) 

The first step in creating a bag of words model is tokenization. In tokenization, we break a sentence up into disconnected words.
Here's a diagram in which we tokenize the two sentences we mentioned above:
![tokenizing-the-headlines2.svg](/images/posts/2018_3_31/tokenizing-the-headlines2.svg)    
As you can see, all we're doing is splitting each sentence into a list of individual words, or tokens. The split occurs on the space character (" ").

- Split each headline into individual words on the space character(" "), and append the resulting list to tokenized_headlines. 

- When you're finished, tokenized_headlines should be a list of lists. Each list should contain the tokens for the headline located at the - - corresponding position in the submissions dataframe.


```python
tokenized_headlines = []

for row in submissions["headline"]:
    tokenized_headlines.append(list(row.split(' ')))
```

We now have tokens, but we need to process them a bit to make our predictions more accurate. We know that Berlin, Berlin., and berlin all refer to the same word, but the computer doesn't know that. We'll need to convert those variations so that they're consistent.

We can do this by lowercasing (which will convert Berlin to berlin), and also by removing punctuation (so Berlin. becomes Berlin).

![preprocessing-tokens-to-increase-accuracy.svg](/images/posts/2018_3_31/preprocessing-tokens-to-increase-accuracy.svg)

Preprocessing doesn't have to be perfect, but the more we can help the computer group the same word together, the higher our prediction accuracy will be. Take a look through your tokens, and see if there are any instances of the same word that you haven't grouped together.

Loop through each item in tokenized_headlines, which is a list of lists.
For each list of tokens:
- Convert each individual token to lowercase
- Remove all of the items in the punctuation list from each individual token
- Append the clean list to clean_tokenized

clean_tokenized should now be a list of lists. Each list should contain the preprocessed tokens associated with the headline in the corresponding position of the submissions dataframe.


```python
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []

for item in tokenized_headlines:
    tokens = []
    for token in item:
        token = token.lower()
        for punc in punctuation:
            token = token.replace(punc, '')
        tokens.append(token)
    clean_tokenized.append(tokens)
```

Now that we have our tokens, we can begin converting the sentences to their numerical representations. First, we'll retrieve all of the unique words from all of the headlines. Then, we'll create a matrix, and assign those words as the column headers. We'll initialize all of the values in the matrix to 0.
![assembling-a-matrix-of-unique-words.svg](/images/posts/2018_3_31/assembling-a-matrix-of-unique-words.svg)

We'll use a pandas dataframe instead of a NumPy matrix. We can create a dataframe with all zero values using this syntax: 


    pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

The code above will create a dataframe with as many rows as the number of items in clean_tokenized. Each column name will be a word from unique_tokens. This assumes that we already assigned the unique tokens to unique_tokens. Each cell in the dataframe will have the value 0. You can find more documentation on initializing a dataframe in [the pandas documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

Find all of the unique tokens in clean_tokenized, and assign the result to unique_tokens.

- Only add tokens that occur more than once (across all of the headlines). Tokens that only occur once don't add anything to the model's prediction power, and removing them will make our algorithm run much more quickly.

- To do this, you can keep a list of the tokens that occur once in the data, and a different list of the tokens that occur more than once. If a token is already in the first list when you encounter it and it's not in the second list, you should add it to the second list.

- When you're finished, unique_tokens should contain any tokens that occur more than once across all of the headlines.

- Each token in unique_tokens should only appear in the list a single time.

Create a dataframe with as many rows as there are items in the clean_tokenized list. Each column name should be a token in unique_tokens. Initialize all of the cells to the value 0. Assign the dataframe to the variable counts.


```python
import numpy as np
unique_tokens = []
single_tokens = []

for tokens in clean_tokenized:
    for token in tokens:
        if token not in unique_tokens:
            single_tokens.append(token)
        if token in single_tokens and token not in unique_tokens:
            unique_tokens.append(token)
        
counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)
```

Now that we have a matrix where all values are 0, we need to fill in the correct counts for each cell. This involves going through each set of tokens, and incrementing the column counters in the appropriate row.

![counting-token-occurrences.svg](/images/posts/2018_3_31/counting-token-occurrences.svg)

When we're finished, we'll have a row vector for each headline that tells us how many times each token occured in that headline.

To accomplish this, we can loop through each list of tokens in clean_tokenized, then loop through each token in the list and increment the proper cell.

- Loop through each list of tokens in clean_tokenized. 
You should use the enumerate() function when writing the loop to get an index along with the list of tokens.
- Loop through each token in the list of tokens.
- Check whether the token is in unique_tokens. If not, it isn't a column in the dataframe, and you should ignore it.
- Increment the appropriate cell by indexing the row of counts, and finding the right column for the token. Add 1 to the cell to indicate that you found the token once.


```python
for i, tokens in enumerate(tokenized_headlines):
    for token in tokens:
        if token in unique_tokens:
            counts.iloc[i][token] += 1
        
```


```python
counts.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>software</th>
      <th>sadly</th>
      <th>we</th>
      <th>did</th>
      <th>adopt</th>
      <th>from</th>
      <th>the</th>
      <th>construction</th>
      <th>analogy</th>
      <th></th>
      <th>...</th>
      <th>bzier</th>
      <th>curves</th>
      <th>headshots</th>
      <th>telephones</th>
      <th>accessories</th>
      <th>rotjs</th>
      <th>roguelike</th>
      <th>nissan</th>
      <th>connecting</th>
      <th>response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 6918 columns</p>
</div>



We have over 2000 columns in our matrix. This can make it very hard for a linear regression model to make good predictions. Too many columns will cause the model to fit to noise instead of the signal in the data. 

There are two kinds of features that will reduce prediction accuracy. Features that occur only a few times will cause overfitting, because the model doesn't have enough information to accurately decide whether they're important. These features will probably correlate differently with upvotes in the test set and the training set.

Features that occur too many times can also cause issues. These are words like and and to, which occur in nearly every headline. These words don't add any information, because they don't necessarily correlate with upvotes. These types of words are sometimes called stopwords.

To reduce the number of features and enable the linear regression model to make better predictions, we'll remove any words that occur fewer than 5 times or more than 100 times.


- Generate a vector that contains the sum of each column in counts. This data will indicate how many times each word occurs in the headlines. You can use the sum() method on pandas dataframes to accomplish this. Assign this vector to word_counts.

- Use the vector to filter counts to remove any columns that occur less than 5 times, or more than 100 times. You can use the loc method on dataframes to accomplish this.


```python
word_counts = counts.sum(axis=0)
```


```python
counts = counts.loc[:, (word_counts <= 100) & (word_counts >= 5)]
counts.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>software</th>
      <th>we</th>
      <th>from</th>
      <th>more</th>
      <th>tool</th>
      <th>released</th>
      <th>de</th>
      <th>not</th>
      <th>as</th>
      <th>good</th>
      <th>...</th>
      <th>6</th>
      <th>product</th>
      <th>y</th>
      <th>add</th>
      <th>user</th>
      <th>things</th>
      <th>local</th>
      <th>skills</th>
      <th>14</th>
      <th>wrong</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 235 columns</p>
</div>



Now we'll need to split the data into two sets so that we can evaluate our algorithm effectively. We'll train our algorithm on a training set, then test its performance on a test set.

The [train_test_split()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from scikit-learn will help us accomplish this.

We'll pass in .2 for the test_size parameter to randomly select 20% of the rows for our test set, and 80% for our training set.
X_train and X_test contain the predictors, and y_train and y_test contain the value we're trying to predict (upvotes).


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)
```

Now that we have a training set and a test set, let's train a model and make test predictions. We'll use a linear regression algorithm from scikit-learn, which you can read more about in the [scikit-learn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

First we'll initialize the model using the LinearRegression class. Then, we'll use the fit() method on the model to train with X_train and y_train. Finally, we'll make predictions with X_test.

When we make predictions with a linear regression model, the model assigns coefficients to each column. Essentially, the model is determining which words correlate with more upvotes, and which with less. By finding these correlations, the model will be able to predict which headlines will be highly upvoted in the future. While the algorithm won't have a high level of understanding of the text, linear regression can generate surprisingly good results.

- Train clf using the fit() method.
- Use the predict() method on clf to make predictions on X_test. Assign the result to predictions.


```python
from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
```


```python
mse = np.sum((predictions - y_test) ** 2) / len(predictions)
```


```python
mse
```




    2135.3631014155517



Our MSE is 2181, which is a fairly large value. There's no hard and fast rule about what a "good" error rate is, because it depends on the problem we're solving and our error tolerance. 

In this case, the mean number of upvotes is 10, and the standard deviation is 39.5. If we take the square root of our MSE to calculate error in terms of upvotes, we get 46.7. This means that our average error is 46.7 upvotes away from the true value. This is higher than the standard deviation, so our predictions are often far off-base.

We can take several steps to reduce the error and explore natural language processing further. Here are some ideas for your next steps:

- Use the entire data set. While we used samples in this mission, you could download the entire data set from this [GitHub repository](https://github.com/arnauddri/hn). This approach will reduce the error rate dramatically. There are many features in natural language processing. Using more data will ensure that the model will find more occurrences of the same features in the test and training sets, which will help the model make better predictions.
- Add "meta" features like headline length and average word length.
- Use a random forest, or another more powerful machine learning technique.
- Explore different thresholds for removing extraneous columns.


```python
np.sqrt(mse)
```




    46.209989195146449


