'''
Created on Mar 28, 2017

@author: Ramanpreet Singh
        b4s79@unb.ca
        
Task: Training word2Vec Model using the set of text files.
Main method to run : trainWord2VecModel()

'''
import os
import glob
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

# Gets user home path. 
userhome = os.path.expanduser('~')

# Training files path and modelNames
modelName='Word2VecModelChDicken'
word2VecModelPath=userhome +r'/Desktop/data/trainModels/{mName}'.format(mName=modelName)
trainCorpusPath=userhome +r'/Desktop/data/trainSet/'

# Word2Vec Model Parameters
pWindow=5
minCount=1
skgramOrCBOW=1# 1 for SkipGram and 0 for CBOW


'''Main Method to train Word2Vec model on a set of training .txt files. 
  Takes pWindow, skgramOrCBOW, minCount as model parameters.
  Upon successful training, Saves the model on the local Disk.  

'''
def trainWord2VecModel():
    print('Starting the Word2Vec Model Training....')
    print('   ')
    
    # Get all the file names in the folder for Training the Sentences. 
    allFilesList=glob.glob(trainCorpusPath+'*.txt')
    
    print('Total Files to Train from '+ str(len(allFilesList)))
    
    # Iterate over all the files and get the raw content.
    sentences=[]
    for singleFile in allFilesList:
        inputFile = open(singleFile, 'r', encoding='utf-8', errors='ignore')
        # Full text
        docText= inputFile.read()
        
        # Get all the Sentences
        allSentences=sent_tokenize(docText)
        count =1
        for t in allSentences:
            #Gets all Tokens in a sentence, clean it for Punctuations and lower case everything.
            allTokens=word_tokenize(t)
            cleanedTokens=[w.lower() for w in allTokens if w.isalpha()]
            count=len(cleanedTokens)+count
            sentences.append(cleanedTokens)
        print('Training from File '+ singleFile+ ' Total Sentences: '+str(len(allSentences))+ ' Total Tokens: '+str(count) )   #print (allTokens)
    
    model = Word2Vec(sentences, min_count=minCount, sg=skgramOrCBOW, window= pWindow)
    
    print('Model Trained on ' + str(len(allFilesList))+" Files")
    print("Saving the Model to "+ word2VecModelPath)
    model.save(word2VecModelPath)



# Run this function to perform training.
trainWord2VecModel()
