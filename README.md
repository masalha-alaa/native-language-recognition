
# Native Language Recognition
This project is an attempt to implement the integral part of the paper **Native Language Cognate Effects on Second Language Lexical Choice** [[1]](#1), in which the authors try to distinguish the native languages of Reddit posters, relying on their Reddit posts which were written in English, using Machine Learning (ML) and Natural Languages Processing (NLP).

Although the authors have released both their dataset and [cleanup code](https://github.com/ellarabi/reddit-l2) publicly, I've created this project from scratch, including fetching (and cleaning) the reddit posts using the PRAW library [[3]](#3) by myself.
The choice of the subreddits to fetch the data from is based on several aspects, such as mother tongue variety and, most importantly, labels. The authors have picked subreddits that the user **flair tag** is almost guaranteed<sup>1</sup> to be their country of origin (e.g. Germany / US / Spain / etc.), so they could use them as the ground truth labels.
It's worth mentioning, however, that the authors' dataset is much larger than mine (8GB vs. 111MB). The size differences are due to several reasons:
1. I have used only 2 subreddits (r/AskEurope, r/Europe), whereas the authors have used an additional 3: r/EuropeanCulture, r/EuropeanFederalists r/Eurosceptics.
2. I was limited by PRAW's 1000 items limitation.
3. The authors have used a clever technique of database expansion, by mining the redditors' personal public profiles and collecting their posts even from outside the 5 main subreddits, after validating their country of origin.
4. I don't have enough resources to process such a big database.

<sup>1</sup>*"Almost guaranteed" because there might be some noise, but the authors have validated this information in a very high confidence using several ways, which are mentioned in this paper and their other paper [[2]](#2).*

## Running the code
In order to run the code, please follow the following steps:

0. Install Python.
1. Clone the repository.
2. Run `install_dependencies.bat` (Windows) or `install_dependencies.sh` (Linux) to setup the virtual environment and install the required libraries.
<br>From now onward, you should be running the scripts from inside the virtual environment.
<br>*If you want to use the existing database, skip to step 9.*<br>
3. Create a reddit app at: [reddit authorized apps](https://www.reddit.com/prefs/apps).
4. Run the script `fetch_data.py` with your app's client ID, secret, and your user agent:<br>
`python fetch_data.py --id <app id> --secret <app secret> --agent <user agent>`
5. Manually remove unwanted noise files, and merge files denoting the same country (e.g. "USA", "US", "United States").
6. Run `preprocessing.py`.
7. If you want to see some data visualization graphs such as sentences average and median lengths etc., run `data_visualization.py`.
8. Run `features_bldr.py`.
9. Run `classfication.py`. The results path will be printed in the console at the end.

## The Authors' Results
The authors have divided their work to 3 tasks:

1. **Binary Nativity Classification:** Distinguish native vs. non-native English speakers.
2. **Language Classification:** Detect the native language of the original poster, among 31 different languages <sup>2</sup>.
3. **Language Family Classification:** Classify posts according to the poster's language family. Language families are:

**English:** Australia, UK, New Zealand, US, Canada.

**Germanic:** Austria, Denmark, Germany, Iceland, Netherlands, Norway, Sweden.

**Romance:** France, Italy, Mexico, Portugal, Romania, Spain.

**Slavic:** Bosnia, Bulgaria, Croatia, Czech, Latvia, Lithuania, Poland, Russia, Serbia, Slovakia, Slovenia, Ukraine.

<sup>2</sup>*Actually it's not clear to me whether it was 31 or 39+6 (45) from the paper. I went on with 31.*

The authors report the following accuracy results for tasks 1,2,3 respectively: 90.8%, 60.8%, 82.5% (using 10 fold cross validation).<br>
It's also worth mentioning that the authors have relied solely on lexical / syntactic features, opposed to context and social network features which they have mentioned in their other paper [[2]](#2). For example some of the features they mention are part of speech (POS) ngrams, function words, and most common words in the database. This is important mainly because the task of the paper (as its title implies) is to recognize the cognate effects of the posters' first language (L1) that are reflected on their second language (L2). I.e. they hypothesize that the effect of L1 is so powerful (in means of word and grammar choice) that it's clearly reflected on L2. For example (from my findings), it turns out that Polish people use the combination "IN JJ NN" ('Preposition' 'Adjective' 'Singular Noun') way more often than native English speakers do, and "NN IN DT" trigrams ('Singular Noun' 'Preposition', 'Determiner') way less. An example of the former is: _"in old building"_ in the sentence:
> "Mainly because most people don't trust the water quality in Poland nor the water pipes since most people live ***in old building*** and they are scared that they are somehow polluted"


Hereby I try to reproduce the authors' results with my own model.
Before starting, it's worth mentioning that the task of native language identification from reddit posts would not be an easy one for a human (rather than a computer), since according to the authors' findings, the non-native reddit posters are at a near-native English level (even higher than advanced TOEFL learners). This assesment was done using various measures such as type-to-token ratio (TTR), average age-of-acquisition of lexical items, and mean word rank, but it's also a well known fact among redditors:

![image](https://user-images.githubusercontent.com/78589884/123857243-bac10d80-d92a-11eb-9d0a-e044f61c5d76.png)


### My Results TLDR
My own results for Tasks 1,2,3 are: 93%, 57.7%, 77.3% respectively.

---

### Longer Walkthrough
Following is a more comprehensive walkthrough of the classification process, which was done using a Logistic Regression Classifier in all the tasks.

#### Task 1 - Binary Nativity
First, for a sanity check, I used a Logistic Regression classifier with the binary occurrences of the top 1000 words used in the database as a feature vector. Namely, my feature table at cell **\[i,j\]** contained `True` if word **j** appeared in text **i** that's being classified, and `False` otherwise. Expectedly, the results were very high (97% accuracy using 10 fold cross validation), affirming the soundness of the database:

<img src="https://raw.githubusercontent.com/masalha-alaa/native-language-detection/master/docs/images/cm%20binary%20nativity%201k%20words.png" alt="Binary Nativity Confusion Matrix, 97% accuracy" width="666">

Although this has achieved a very high accuracy, it is an expected and not a very interesting result. As mentioned earlier, the main interest here is to perform the task by relying only on semantic features. Thus, I used the same Logistic Regression classifier, but this time with different features: the corpus's top 1000 POS trigrams, and a list of widely used English function words (for example see [[4]](#4) for common English function words). The results were quite satisfying as well (about the same as the authors'):

<img src="https://github.com/masalha-alaa/native-language-detection/blob/master/docs/images/cm%20binary%20nativity%20fw%20%26%201k%20pos.png" alt="Binary Nativity Confusion Matrix, 93% accuracy" width="666">

#### Task 2 - Language Classification
Once again, using the top 1000 words binary features, I achieved very good results (87% accuracy). Note that a baseline random chance classifier would have a **1/24=4%** accuracy.

<img src="https://github.com/masalha-alaa/native-language-detection/blob/master/docs/images/cm%20country%20identification%201k%20words.png" alt="Country Confusion Matrix, 1K words, binary, 87% accuracy, 93% accuracy" width="666">

As mentioned earlier, these results are not interesting, but merely serve as a sanity check for the database. However, it might be worthy to examine the most important features determined by the classifier (**click the image for an interactive plot _(produced by PlotLy)_**):

[![Heatmap - country 1k words tfidf](https://github.com/masalha-alaa/native-language-detection/blob/master/docs/images/hm%20country%20identification%201k%20words.png)](https://rawcdn.githack.com/masalha-alaa/native-language-recognition/ea009af7ad125bb2ac2cc5d8b0201165ef632f84/docs/2021-05-22%2018-25-01%20hm.html)

This is a column-wise heatmap, in which the rows are classes (countries) and the columns are features selected as "best features" by the classifier, so that cell **\[i,j\]** is how many times the word **j** appeared in documents of class **i**. Being a column-wise heatmap, it grants us the benefit of easily identifing each feature's distribution across the different countries. For example we can see that the word "austria" is mostly used by Austrian authors, whereas the word "german" by German authors, and so on. This strengthens the assumption that people tend to mention their own culture and countries more often than others.

Moving on to the more interesting semantic-features results, here is the confusion matrix of country classification, using only common function words and the top 1000 POS trigrams:

<img src="https://github.com/masalha-alaa/native-language-detection/blob/master/docs/images/cm%20country%20identification%20fw%20%26%201k%20pos.png" alt="Country Identification FW & 1K POS TRI 53.7% accuracy" width="666">

As can be seen in the confusion matrix above, I have achieved a satisfying accuracy of 53.7% (authors' is 60.8%). As explained earlier, the random chance baseline accuracy would be 4%, so 53% and 60% are very high accuracies compared to that ([click to see an interactive heatmap of the best features selected by the classifier _(produced by Plotly)_](https://rawcdn.githack.com/masalha-alaa/native-language-recognition/9d055ede75b1da92bb42bb5c05baa4ef7c40fa2f/docs/2021-05-22%2018-32-37%20hm.html)).
Also, notice that most errors occur between "close countries" (countries in the same language family) - such as all the English speaking countries, or the Romance-language countries, which leads us to the 3rd and final classification task.

#### Task 3 - Language Family Classification
In this task I try to classify the languages into 4 language families: [English](English "Australia, UK, New Zealand, US, Canada") / [Germanic](Germanic "Austria, Denmark, Germany, Iceland, Netherlands, Norway, Sweden") / [Romance](Romance "France, Italy, Mexico, Portugal, Romania, Spain") / [Slavic](Slavic "Bosnia, Bulgaria, Croatia, Czech, Latvia, Lithuania, Poland, Russia, Serbia, Slovakia, Slovenia, Ukraine") _(hover to see details - link is not active)._

Using the same features that I had used in the other 2 tasks (function words and POS trigrams), I achieved a 77.3% accuracy:

<img src="https://github.com/masalha-alaa/native-language-detection/blob/master/docs/images/cm%20family%20identification%20fw%20%26%201k%20pos.png" alt="Family Identification FW & 1K POS TRI 77.3% accuracy" width="666">


### Summary
In this project I have successfully reproduced the authors' results in identifying the original posters' native languages, using NLP techniques. The small differences in the results (in favor of the authors) might be due to various reasons, including but not limited to:
1. Database size (theirs is much larger).
2. Database cleaning (for example, they replaced URLs with the URL token, whereas I simply removed them; and replaced non-English words with the UNK token, whereas I treated them as normal tokens).
3. Their token tagging was more accurate, as a result of using Spacy[[5]](#5) for named entities identification, as well as the 'truecasing' technique to distinguish between abbreviations and regular words such as 'US' (United States) and 'us' (pronoun).
4. The authors might have unified the English speaking countries into 1 class in the Language Classification task (Task 2). I did not do that because unfortuntaely I could not determine from the paper if they have actually done that or not. But obviously if I do this unification the results would be significantly higher in the Language Classfication task.

### Discalimer
I personally know the authors of the paper, and they have guided me through a simplified version of this project back in my MSc studies, and now 4 years later as I'm more skillful and advanced in the ML field and in programming in general, I have decided to create it by my own from scratch.

## Resources
<a id="1">[1]</a>
Rabinovich, Ella, Yulia Tsvetkov, and Shuly Wintner. "Native language cognate effects on second language lexical choice." _Transactions of the Association for Computational Linguistics_ 6 (2018): 329-342.
Available at: <https://transacl.org/ojs/index.php/tacl/article/view/1403>.

<a id="2">[2]</a>
Goldin, Gili, Ella Rabinovich, and Shuly Wintner. "Native language identification with user generated content." _Proceedings of the 2018 conference on empirical methods in natural language processing_. 2018.
Available at: https://www.aclweb.org/anthology/D18-1395/

<a id="3">[3]</a>
PRAW, The Python Reddit API Wrapper.
Available at: https://praw.readthedocs.io/en/latest/

<a id="4">[4]</a>
Function words list: https://semanticsimilarity.files.wordpress.com/2013/08/jim-oshea-fwlist-277.pdf

<a id="5">[5]</a>
Spacy: Industrial-Strength Natural Language Processing.
Available at: https://spacy.io/
