# Language-Modelling-approach-to-Article-Usage-Prediction
Language-Modelling-approach-to-Article-Usage-Prediction

Problem Statement: Given a text, predict the correct usage of article "a" and "the" in an obfuscated text and compare it with the original text. 

Two approaches to solve the problem of Article usage in english text.

1. Language modelling approach 
2. Word Embedding (Word2Vec) Approach

Evaluation Metrics: The final evaluation is preformed using standard Precision, Recall  and F1 measures. 

Data is provided into ~/data folder.
1. ~/trainSet contains 19 Charles Dickens writings. Taken from  <http://www.textfiles.com/etext/AUTHORS/DICKENS/>
2. ~/testSet contains the file to be test.
3. ~/trainModels contains the trained word2Vec model.
4. ~/originalSet contains the original version of file to be tested


External Dependencies: 
1. nltk for tokenization, sentence segmentation and corpora building
2. gensim for word2Vec word embedding
3. sklearn for evaluation metrics


Running Instructions:

1. Save all the Data from ~/data folder to the Desktop. 
2. Update all the path information in ArticleUsagePrediction.py and TrainWord2Vec.py  with the location of ~/data folder.
3. Now Run TrainWord2Vec.py and this will populate word2Vec Model file in folder ~/data/trainModels/ by the name of "Word2VecModelChDicken"
4. Run ArticleUsagePrediction.py. It will load the populated model and perform prediction on testData stored in ~data/testSet/

