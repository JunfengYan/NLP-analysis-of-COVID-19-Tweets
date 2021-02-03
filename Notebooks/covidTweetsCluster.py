# Databricks notebook source
# MAGIC %md ##Setting up environment and importing packages

# COMMAND ----------

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import math


%matplotlib inline

# COMMAND ----------

datapath = "dbfs:/FileStore/tables/covid_train_hydrated.csv"

# COMMAND ----------

# MAGIC %md ## reading and cleaning the data

# COMMAND ----------

tweets_Df =  spark.read.format("csv").option("header", "true").load(datapath)

# COMMAND ----------

#dropping null rows
from pyspark.sql import functions as F

tweets = tweets_Df.select("id","text")

def pre_process(df):
  df_drop = df.filter(df['text'].isNotNull())
  df_drop = df_drop.filter(df_drop['id'].isNotNull())
  df_drop = df_drop.dropDuplicates()
  
  print('After dropping, we have ', str(df_drop.count()), 'row in dataframe')
  return df_drop

tweets_drop = pre_process(tweets)




# COMMAND ----------

#dropping text rows that has "false" as value
tweets_drop = tweets_drop.where(~tweets_drop.text.like('false'))
tweets_drop.count()

# COMMAND ----------

#used regular expression matching to get rid of urls

tweets_clean = tweets_drop.select('id', (F.lower(F.regexp_replace('text', "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "")).alias('text')))


# COMMAND ----------

#total number of tweets
tweets_clean.count()

# COMMAND ----------

# MAGIC %md ##Building the pre-processing pipeline

# COMMAND ----------

#spark nlp does not comes with collection of stop words
#so I used the English stop words from the nltk library
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append("amp")

print ("We use " + str(len(stopwords)) + " stop-words from nltk library.")
print (stopwords[:10])

# COMMAND ----------

#building text pre processing pipeline with Spark NLP and Spark ML

import sparknlp

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from pyspark.ml import Pipeline

#document assembler transform raw text into annotator type, common object type used in spark nlp anootator interface

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence") \
    .setUseAbbreviations(True)

#tokenizer transform sentences into individual words for each tweet

tokenizer = Tokenizer() \
    .setInputCols(['sentence']) \
    .setSplitPattern("\p{P}(?<!')") \
    .setMinLength(3)\
    .setMaxLength(16)\
    .setOutputCol("token")

#Removes all dirty characters from text following a regex pattern and transforms words based on a provided dictionary
#this step is probably redundant here since we already did the text cleaning above

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")\
    .setLowercase(True)\
    .setCleanupPatterns(["[^\w\d\s]"])

#drop all stop words i.e. words with no inherent meaning in the list of strings
stopwords_cleaner = StopWordsCleaner() \
        .setInputCols(["normalized"]) \
        .setStopWords(stopwords)\
        .setCaseSensitive(False)\
        .setOutputCol("removed")

#retrieving the meaningful part of the word i.e. stem without prefix or suffixes
#stemmer = Stemmer() \
    #.setInputCols(["removed"]) \
    #.setOutputCol("stem")

# using a dictionary or through morphological analysis to return words to their base form
lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['removed']) \
     .setOutputCol('lemma')

#generate bi-grams and tri-grams from the lemmatized tokens
ngrams_cum = NGramGenerator() \
            .setInputCols(["lemma"]) \
            .setOutputCol("ngrams") \
            .setN(3) \
            .setEnableCumulative(True)
#produced processed text data
finisher = Finisher() \
    .setInputCols(["removed","lemma","ngrams"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(True)

#assembling the pipeline
nlpPipeline = Pipeline(stages=[document_assembler,sentence_detector,tokenizer,normalizer,stopwords_cleaner,lemmatizer,ngrams_cum,finisher])                    

# COMMAND ----------

#fitting the model and transforming the data
mod = nlpPipeline.fit(tweets_clean)
cleaned = mod.transform(tweets_clean)

# COMMAND ----------

cleaned.select('finished_lemma','finished_ngrams').show(10,truncate = False)

# COMMAND ----------

#using regex matching to make different ways of naming coronavirus the same
training = cleaned.select("id",F.explode("finished_ngrams").alias("elements"))\
          .select("id",F.regexp_replace("elements",'(\w*corona\w*)|(\w*covid\w*)',"coronavirus").alias("elements"))\
          .groupby("id")\
          .agg(F.collect_list("elements").alias("ngrams")) 
        

# COMMAND ----------

training.select("ngrams").show(5,truncate = False)

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF

#obtaining tf-idf matrix
TF = CountVectorizer(inputCol="ngrams", outputCol="rawFeatures",minDF=0.01,maxDF=0.99,vocabSize = 1000)
tf_model = TF.fit(training)
featurizedData = tf_model.transform(training)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# COMMAND ----------

#our bag of words
vocab = tf_model.vocabulary
vocab

# COMMAND ----------

len(vocab)

# COMMAND ----------

# MAGIC %md  ## trainning k-means ++ model

# COMMAND ----------

#training with all 142 vocabulary
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()
#Optimize choice of k
#for each k, use a different 10% sample to fit and compute wssse
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(565).setFeaturesCol("features").setInitSteps(10)
    model = kmeans.fit(rescaledData.sample(False,0.1, seed=42))
    pred = model.transform(rescaledData)
    cost[k] = evaluator.evaluate(pred)

# COMMAND ----------

plt.figure(dpi=1200)
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20], linestyle='--', marker='o')
ax.set_xlabel('k')
ax.set_ylabel('cost')
ax.set_title('Silhouette for k from 2 to 20')
for x,y in zip([6,7,9,12],[cost[6],cost[7],cost[9],cost[12]]):
  xlab = "{:.2f}".format(x)
  plt.annotate(("K = " + xlab),(x,y),textcoords="offset points",xytext = (0,5),ha = 'center')
plt.savefig('this.png')
display()

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

best_K_means = KMeans().setK(7).setSeed(565).setFeaturesCol("features").setInitSteps(10)
best_model = best_K_means.fit(rescaledData)
predictions = best_model.transform(rescaledData)

# COMMAND ----------

from pyspark.ml.clustering import BisectingKMeans

alt_K_means = BisectingKMeans().setK(4).setSeed(565).setFeaturesCol("features").setDistanceMeasure('cosine')
alt_model = alt_K_means.fit(rescaledData)
predictions = best_model.transform(rescaledData)


# COMMAND ----------

dir(best_model.summary)

# COMMAND ----------

alt_model.summary.clusterSizes

# COMMAND ----------

centers = alt_model.clusterCenters()

# COMMAND ----------

def printCenters1(centers):
  for i in range(len(centers)):
    print ("Cluster " + str(i) + " words:", end='')
    order_centroids = centers[i].argsort()[::-1]
    for ind in order_centroids[:10]:
        print(vocab[ind] + ",",end = '')
    print()

# COMMAND ----------

len(centers)

# COMMAND ----------

printCenters1(centers)

# COMMAND ----------

def getCenters1(centers):
  centers_summary = {}
  for i in range(len(centers)):
    summary = []
    order_centroids = centers[i].argsort()[::-1]
    for ind in order_centroids[:20]:
       summary.append(vocab[ind])
    centers_summary[i] = summary
  return centers_summary

# COMMAND ----------

summary = getCenters1(centers)


# COMMAND ----------

v = set()

# COMMAND ----------

def findIdenticalWords(summary):
  out = []
  for i in range(len(summary)):
    xx = set(summary[i])
    out.append(xx)
  u = set.intersection(*out)
  
  return u
  
u = findIdenticalWords(summary)

# COMMAND ----------

u

# COMMAND ----------

best_K_means = KMeans().setK(6).setSeed(565).setFeaturesCol("features").setInitSteps(10)
best_model = best_K_means.fit(rescaledData)
predictions = best_model.transform(rescaledData)

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
silhouette

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC using lower dimensionality to see if we have better results?

# COMMAND ----------

#limit the dictionary to only top 30 
TF_small = CountVectorizer(inputCol="ngrams", outputCol="rawFeatures",minDF=0.01,maxDF=0.99,vocabSize = 30)
tf_model_small = TF_small.fit(training)
small_features = tf_model_small.transform(training)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
smallIdf = idf.fit(small_features)
small_data = smallIdf.transform(small_features)

# COMMAND ----------

tf_model_small.vocabulary