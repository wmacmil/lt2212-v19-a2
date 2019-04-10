# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Warrick Macmillan

## Additional instructions

see myscript.sh and ms2.sh to see the commmand line code run

## File naming convention

See myscript.sh for the input parameters, but they just occur in the same order as the files listed.

### Vocabulary restriction.

I gave 1500 word vocab restriction, because it was significantly smaller than the raw vocabulary size but was presumably large enough to capture a somewhat finer level of understanding.  This hypothesis seems to be somewhat contracted by the only roughly 3% improvement.

### Result table

[(second,first)     , (first, second)    , (first,first)      , (second, second)]
[0.30357529885236334, 0.30357529885236334, 0.3240299216019181, 0.36531232564334276]
[0.32862536733396525, 0.3286253673339653, 0.3492089604886903, 0.39654289470942644]
[0.06872875265934303, 0.06872875265934304, 0.09403037364003336, 0.10222264718406482]
[0.10033975568893838, 0.10033975568893841, 0.13108127810035142, 0.1486086201759006]
[0.4178184611383674, 0.4178184611383673, 0.45744361558047464, 0.4984540257419908]
[0.30440337280563373, 0.3044033728056338, 0.3254864792156253, 0.3662826764078078]
[0.16868623213183562, 0.16868623213183562, 0.21708604584884691, 0.24963759651007444]
[0.06935471790835548, 0.06935471790835553, 0.09511954940612688, 0.10320479609964142]

where first and second refer to the respective corpuses

### The hypothesis in your own words

I don't really didn't have an a priori hypothesis, but I suppose the hypothesis was that dimension reductionality would improve the results because SVD is a way of producing a more representative vector space than the large dimension one spanned by the one hot vectors.  Additionally, I remember seeing war or some kind of violence-like word was used almost exclusively in one of the corpuses.  I suppose this means that certain topics were definitively covered more in the corpuses.  It would be interesting to somehow categorize the frequent words to get an understanding of the differences.

I didn't expect the tf-ifd to decrease the cossine similarity, but this makes a posteri sense because the less more frequent words shouldn't outwiegh the less frequent ones in the comparison.

### Discussion of trends in results in light of the hypothesis

Important things to note in the data below:

second == fine
first == coarse

Cossim is billinear and hence equal to its transpose ((first,second) == (second,first)).
The both corpuses were uniformly more similair to themselves than to each other, and the second was uniformly more self similair than the first.
Tfidf uniformly reduced the cosine similarity.
SVD uniformly increased cosine similarity.

## Bonus answers

My answer is somewhat mixed, but the flaws are more or less contained in the first half and imporvements/suggestions generally occur later.

I'll give two examples of utterences with the same vector representations and therefore are treated the same by the model.  

  'I love Donald Trump.  I hate Hillary Clinton.'
  'I love Hillary Clinton. I hate Donald Trump.'

would have the same vector represenations, despite having "opposite" meanings.  In a different instance: 

  "The dog watched a man baking the cake." 
  "A man baking the dog watched the cake."

which have completely independent, unrelated meanings would both receive the same vector representations in our model.

One could imagine any number of longer documents which provide even greater absurdities arising from the bag of words assumptions.  

Other problems include time and space complexity.  The naive approach works because of modern computational processing, but it seems like preprocessing and optimizing would need to be addressed to scale this techique to larger data sets.  Obviously word2vec can significantly reduce vector dimensionality.

Additionally, when discussing this to a class at stanford, Chris Manning says about one hot encodings :

  "no inherent notion of relationship between words"
  "onehot encodings have no natural notion of similarity"

e.g. no natural semantic's exist b/c all vectors are orthogonal

Other improvements to be made could use syntactic and semantic representations to significantly extract more information through which various documents could be compared, and then somehow compare the parse trees or other data representations of the texts with some notion of meaningful "closeness".  This is an ongoing, even upcoming research topic, see [1]. One could also try using a different language model (e.g. ngrams) to possibly get more natural comparison that just a pure bag of words approach.


[1] Learning Structured Natural Language Representations for Semantic Parsing (https://arxiv.org/abs/1704.08387)


