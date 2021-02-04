# NLP-analysis-of-COVID-19-Tweets

![Alt text](Images/project_pipeline_illustration.png?raw=true "Text Clustering")

### What is this project about?
-This project implemented a pipeline to pre-process, cluster, and extract central rhetorics from  1.2 million tweets that mentioned COVID-19.

### How is it implemented?
-This project is implemented using Databricks notebook and apache spark, and the main package I used is SparkNLP and Spark ML lib.

### Future direction?
-This project is a work in progress. Future direction will focus on better pre-processing of the text(e.g. coding emojis to word instead of getting rid of them) as well as implementing a more sophisticated model (e.g. Word2Vec).

For more details please check out the [main notebook](Notebooks/covidTweetsCluster.py)
