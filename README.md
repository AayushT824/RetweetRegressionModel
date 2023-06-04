# RetweetRegressionModel

I decided to attempt to predict the amount of retweets a particular tweet would receive based on its content and when it was posted. Two models were used here - a k nearest regressor and a feed-forward neural network. Alternative regressors such as support vector, Random Forest and Gaussian process were also attempted and played with, but ended up being unable to train in any reasonable amount of time. The final MAE for both models stood at around 6,000. For reference, a completely uninformed model stood at an MAE of about 20,000.

To simplify the dataset and focus on the relevant features for predicting the
number of retweets, the columns 'id', 'link', 'geo', 'favorites', and 'hashtags were
dropped'. The reasoning was that the number of favorites in a tweet could potentially
reflect the number of retweets and, therefore, should not be included as features. There
is a direct correlation between the number of retweets and the number of favorites a
tweet has gotten since they both cause the popularity of the tweet. The number of
hashtags were considered, however it was deemed that they would not affect tweets
enough and could be unnecessary noise.

Furthermore, the 'date' column was dropped and from it two new columns, 'hour'
and 'month' were extracted. The ‘month column was added to the dataset to capture
any seasonal trends that might affect the number of retweets a tweet would receive.
Since there is a similar idea with posting on Instagram at certain times to get the most
likes these models would see if there was a time relation hence the ‘hour’ column was
added.

The number of @s a tweet had could also be thought to affect the popularity of a
tweet. To see if tagging others in a tweet would affect the number of retweets, this
feature was included. Preprocessing was required to cleanse this feature and substitute
null/invalid values with their intended representation.

In addition to dropping unnecessary columns and extracting new features, I
also preprocessed the dataset by removing outliers and considering additional features.
Specifically, we removed all tweets with less than 25 retweets and all tweets with more
than 75,000 retweets. This was done to prevent extreme values from skewing the
analysis and to focus on tweets that had a more typical level of engagement.

Sentiment analysis was also conducted and used as a feature for each tweet using Textblob. A tokenizer specifically constructed to work with tweets was also selected as opposted to typical word tokenizers, and basic stopword and punctuation data cleansing was also applied.

We then experimented with multiple hyperparamter tunings for the FFNN, eventually noticing that a 256 and 128 neuron layered model worked merginally better than the others, with dropout layers in between.
