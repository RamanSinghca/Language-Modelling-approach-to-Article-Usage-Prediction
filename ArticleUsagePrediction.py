'''
Created on Mar 28, 2017

@author: Ramanpreet Singh
        b4s79@unb.ca
        
Task: Article Usage Prediction in English text Using Language modeling and word2Vec Approach.
Main Method to Run : runPredictionEngine()

'''
import os
import nltk
import random
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn import metrics

# Gets user home path. 
userhome = os.path.expanduser('~')

# Some Files and Model Names.
orgFileName='CharlesDickens_orig'
testFileName='CharlesDickens_obfuscated';
modelName='Word2VecModelChDicken'

# File and Path Names: Change them according to your local settings.
orgfilePath=userhome +r'/Desktop/data/original/{fileName}.txt'.format(fileName=orgFileName)
testFilePath=userhome +r'/Desktop/data/testSet/{fileName}.txt'.format(fileName=testFileName)
trainCorpusPath=userhome +r'/Desktop/data/trainSet/'
word2VecModelPath=userhome +r'/Desktop/data/trainModels/{mName}'.format(mName=modelName)


'''Uses the Original File stored at OrgfilePath and generates the ground truth labels in a list.
    0 means article "a"
    1 means article "the"
    Output: List of Labels [0,0,1,1,1,1,0,0......]
'''
def getGroundTruth():
    groundTruthList=[]
    # Get all the file names in the folder for Training the Sentences. 
    
    inputFile = open(orgfilePath, 'r', encoding='utf-8', errors='ignore')
    docText= inputFile.read()
    allSentences=sent_tokenize(docText)
    count =1
        
    for t in allSentences:
        allTokens=word_tokenize(t)
        count=len(allTokens)+count
        for w in allTokens:
            w=w.lower()
            if w =='a':
                groundTruthList.append(0)
            elif w =='the':
                groundTruthList.append(1) 
    
    print('Total Instances of "a" and "the" in the text '+ str(len(groundTruthList)))
    return groundTruthList


# *****************************************************************************************
# ***********************************  Language Modeling Approach ************************
# *****************************************************************************************


''' Fetches the training corpus for training language model
    It retrieves all the text files contained in trainCorpusPath and adds them to object of NLTK corpus.
    output: An instance of NLTK Corpus.
'''
def getTainingCorpus():
    # Get all the files stored in trainCorpusPath
    trianCorpus = PlaintextCorpusReader(trainCorpusPath, '.*\.txt', encoding='utf8')
    for infile in sorted(trianCorpus.fileids()):
        trianCorpus.open(infile)  
    return trianCorpus

''' BiGram Language Model Creation and Probability calculation.

'''
cFreqBigram = nltk.ConditionalFreqDist(nltk.bigrams(w.lower() for w in getTainingCorpus().words()))
cProbBigram = nltk.ConditionalProbDist(cFreqBigram, nltk.MLEProbDist)


'''Takes Suffix (String word) and generates the probability of sequences P("a","suffix") and P("the","suffix")
     This uses Language Modeling Approach.
     Input: Suffix- String Token
     output:
     0 if Article prediction is "a"
     1 if Article prediction is "the"
'''

def getPredictionByLanguageModel(suffix):
        if cProbBigram["a"].prob(suffix) >= cProbBigram["the"].prob(suffix):
            return 0
        elif cProbBigram["a"].prob(suffix) < cProbBigram["the"].prob(suffix):
            return 1
    
        
# *****************************************************************************************
# ***********************************  WordVec Approach ***********************************
# *****************************************************************************************

'''Load the Trained word2Vec Model from location word2VecModelPath.
    Refer to TrainWord2Vec.py  To get more details on how to train word2Vec Model using text files.  
'''
trainedWord2VecModel = Word2Vec.load(word2VecModelPath)


'''Takes Suffix (String word) and generates the probability of sequences P("a","suffix") and P("the","suffix")
     This uses Word2Vec Approach.
     Input: Suffix- String Token
     output:
     0 if Article prediction is "a"
     1 if Article prediction is "the"
'''
def getPredictionByWord2Vec(suffix):
    
    try:
        if trainedWord2VecModel.similarity('a', suffix) >= trainedWord2VecModel.similarity('the', suffix):
            return 0
        elif trainedWord2VecModel.similarity('a', suffix) < trainedWord2VecModel.similarity('the',suffix):
            return 1
    except KeyError:
        coinToss=random.random()        
        return 0 if 0<coinToss<0.5  else  1


# *****************************************************************************************
# *********************************** Article Usage Prediction ****************************
# *****************************************************************************************

''' Main Method to Run the prediction of article in test file provided by testFilePath
    Input: modelNo Integer
         1 results by BiGram Language model
         2 results by word2Vec Method
         
    Output: List of predictions which is to be matched by groundTruthList generated from original text.
'''
def predictArticle(modelNo):
    print('Starting Prediction Engine on File '+  testFileName+ '.txt')
    predictions=[]
    inputFile = open(testFilePath, 'r', encoding='utf-8', errors='ignore')
    
    docText= inputFile.read()
    # Using Sentence Segmentation Split all the Sentences of Given Document.
    allSentences=sent_tokenize(docText)
    
    for t in allSentences:
        # Get all Tokens in a given Sentence.
        allTokens=word_tokenize(t)
        #Check pair by pair
        for item, next_item in zip(allTokens, allTokens[1:]):
            w=item.lower()
            nextW=next_item.lower()
            if w=='a' or w=='the':
                if (modelNo==1):
                    predictions.append(getPredictionByLanguageModel(nextW))  
                elif (modelNo==2):
                    predictions.append(getPredictionByWord2Vec(nextW))  
    
    print('All the Predictions are made..') 
    print('Next- Evaluation of Results ...')    
    return predictions


# *****************************************************************************************
# *********************************** Results and Evaluation ******************************
# *****************************************************************************************

''' Run this Method for Classification results.
    Input: ModelNo( integer)
         1 for getting results through Language Modeling approach
         2 for getting results through Word2Vec approach.
    
    Output: OnConsole- Precision, Recall and F Measure
        0 means article "a"
        1 means article "the"
'''
def runPredictionEngine(modelNo):
    print('Getting the ground Truth Labels from '+ orgFileName+'.txt')
    groundTruth=getGroundTruth()
    
    if modelNo==1:
        print('Loading the Language Model .... ')
    elif modelNo==2:
        print('Loading the Word2Vec Model from the Disk....')
        
    predicted=predictArticle(modelNo)
    
    print (metrics.classification_report(groundTruth,predicted))

# *****************************************************************************************

''' Execute  this Function 
    Input: ModelNo( integer)
        1 for getting results through Language Modeling approach
        2 for getting results through Word2Vec approach.
    
    Output: OnConsole print- Precision, Recall,  F Measure, Support
        0 means article "a"
        1 means article "the"
'''
runPredictionEngine(1)

