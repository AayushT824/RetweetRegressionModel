# RetweetRegressionModel

I decided to attempt to predict the amount of retweets a particular tweet would receive based on its content and when it was posted. Two models were used here - a k nearest regressor and a feed-forward neural network. Alternative regressors such as support vector, Random Forest and Gaussian process were also attempted and played with, but ended up being unable to train in any reasonable amount of time. The final MAE for both models stood at around 6,000. For reference, a completely uninformed model stood at an MAE of about 20,000.

To simplify the dataset and focus on the relevant features for predicting the
number of retweets, the columns 'id', 'link', 'geo', 'favorites', and 'hashtags' were
dropped. The number of favorites a tweet receives is only discoverable after a tweet is posted, and thus would be impractical to give to our model for help with predictions. The number of
hashtags was considered, but it was deemed that they would not affect retweets
enough and could result in unnecessary noise.

Furthermore, the 'date' column was dropped and from it two new columns, 'hour'
and 'month' were extracted. The 'month' column was added to the dataset to capture
any seasonal trends that might affect the number of retweets a tweet would receive.
A similar approach was taken with the 'hour' column to capture the effect the timing of a tweet may have.

The number of @s a tweet had could also affect the popularity of a
tweet. To see if tagging others in a tweet would affect the number of retweets, this
feature was included. Preprocessing was required to cleanse this feature and substitute
null/invalid values with their intended representation.

In addition to dropping unnecessary columns and extracting new features, we
also preprocessed the dataset by removing outliers and considering additional features.
Specifically, we removed all tweets with less than 25 retweets and all tweets with more
than 75,000 retweets. This was done to prevent extreme values from skewing the
analysis and to focus on tweets that had a more typical level of engagement.

Sentiment analysis was also conducted and used as a feature for each tweet using Textblob. A tokenizer specifically constructed to work with tweets was also selected as opposted to typical word tokenizers, and basic stopword and punctuation data cleansing was also applied.

We then experimented with multiple hyperparamter tunings for the FFNNs eventually concluding that a 256 and 128 neuron layered model worked merginally better than the others, with dropout layers in between.

This project was built off a barebones model I had built prior for the problem. In this project I significantly upgraded the preprocessing strategies and visualizations, as well as improving my feature engineering methods, model construction and hyperparameter tuning. 

Everything I developed in addition to the base model:
- Modifying sample size & experimenting with it to find a good balance between quantity and efficiency for hyperparameter tuning
- Developing Proper noun recognition
    - Before, any words not recognized by the word embedding model were just ignored and not used at all. 
    - Now, each time a proper noun is detected, it is recorded in a proper noun dictionary as a unique 300 length vector
    - It is represented as that vector each time it appears. Used NLTK Part of speech tagging to achieve this.
    - Modified vectorization function to account for above change as well
- Text preprocessing function had to be modified as well and was split from the vectorization application in order to apply functions to the text before it was vectorized, but after it was cleaned by the preprocessing function
- Engineered two new features - links and sentiment score
    - For links, I counted the number of links that appeared in the string representation of the tweet and recorded it
    - For sentiment score, I experimented with building my own sentiment analysis model and training it on one of the many existing kaggle datasets that would permit me to do so, but decided against it because TextBlob, a NLP library had a module that already contained this functionality. Furthermore, my sentiment analysis attempts had run intoa  few roadblocks which I realized would take some time to get around - time I didn't need to spend because I had the functionality I needed already in that module. The module's polarity function generates a number on a scale from -1 to 1 that determines how positive or negative a tweet is, and the degree of extremity.
    - Also experimented with counting the number of positive and negative connotated words in the tweet, using preconstructed lists of words that we know for sure are positiev and negative, but this was also scrapped in favor of the above module
    - The two attempts described in the bullets above are not included here, as this is all just the code that was used in the final product, and those past attempts were thrown away.
- After counting links, removed all URLs and digits from each tweet with the re/replace function
- Switched from using the typical word tokenizer provided by NLTK to their twitter-specific one that was built to tokenize tweets and to handle the anomalies that come with them
- Also using a tweet preprocessor module that removed much of the noise and incomprehensible characters from each tweet. Applied this alongside the vectorization function in the same apply statement.
- Train and testing process was fixed, and the method of adding the engineered features to the test and training sets was completely revamped - previously had just assigned new columns to the existing ones, but switched over to using the pandas join function which prevented the creation of a bunch fo dummy features with all NaN values as the previous method had. Also reset indices to ensure the shapes lined up.
- Experimented with many different hyperparameter combinations for the forward-feeding neural network, from activation functions to layer structure and order, as well as the inclusion of dropout layers.
- When having enough neurons, generally solid performance, but some stood out. 
- Analyzed performance of all tests and produced a final model and trained it for an additional 15 epochs to evaluate its performance.
- Did this same process for the KNr method.
