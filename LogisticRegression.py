# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:16:46 2018

@author: vijendher
"""

import os
import re
import random
import math
import sys
w0 = 0.01
hamPath = os.getcwd() + "\\train\ham"
spamPath = os.getcwd() + "\\train\spam"
testHamPath = os.getcwd() + "\\test\ham"
testSpamPath = os.getcwd() + "\\test\spam"
    
weightsMap = {}
fileWordCountMap = {}
testFileWordCountMap = {}
allWordsSet = set()
listOfWords = []
hamFileSet = set()
testHamFileSet = set()
spamFileSet = set()
testSpamFileSet = set()
allFilesSet = set()
allTestFiles = set()
learningRate = float(sys.argv[2])
fileWeightsMap = {}
newWeightsMap = {}
numberOfIterations = int(sys.argv[3])
lamda = float(sys.argv[4])
useStopWords = sys.argv[1]
stopWords = set()
def main():
    countStop = 0.0
    if useStopWords == 'no':
        toOpenStopWordsFile = open(os.getcwd()+"\\stopwords.txt")
        for line in toOpenStopWordsFile:
            word = line.strip("\n")
            stopWords.add(word)
        
    for hamFile in os.listdir(hamPath):
        openHamFile = hamPath + "\\" + hamFile
        toOpenHamFile = open(openHamFile)
        allFilesSet.add(hamFile)
        hamFileSet.add(hamFile)
        presentFileWordCountMap = {}
        for line in toOpenHamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if not (useStopWords == 'no' and (stripped in stopWords)):
                        if stripped in presentFileWordCountMap:
                            presentFileWordCountMap[stripped] += 1.0
                        else:
                            presentFileWordCountMap[stripped] = 1.0
                        if not (stripped in allWordsSet):
                            allWordsSet.add(stripped)
                            listOfWords.append(stripped)
                    else:
                        countStop = countStop+1
        fileWordCountMap[hamFile] = presentFileWordCountMap
                        
                        
    for spamFile in os.listdir(spamPath):
        openSpamFile = spamPath + "\\" + spamFile
        toOpenSpamFile = open(openSpamFile,encoding='latin-1')
        allFilesSet.add(spamFile)
        spamFileSet.add(spamFile)
        presentFileWordCountMap = {}
        for line in toOpenSpamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if not (useStopWords == 'no' and (stripped in stopWords)):
                        if stripped in presentFileWordCountMap:
                            presentFileWordCountMap[stripped] += 1.0
                        else:
                            presentFileWordCountMap[stripped] = 1.0
                        if not (stripped in allWordsSet):
                            allWordsSet.add(stripped)
                            listOfWords.append(stripped)
                    else:
                        countStop = countStop+1
        fileWordCountMap[spamFile]  = presentFileWordCountMap
    
    
    for word in listOfWords:
        randomWeight = random.random()
        weightsMap[word] = randomWeight
        
        
    fillFileWeightsMap()
            
    train() 
    
    count = 0.0
    hamCount = 0.0
    spamCount = 0.0
    hamMistake = 0.0
    spamMistake = 0.0
    mistake = 0.0
    
    
    for testHamFile in os.listdir(testHamPath):
        openTestHamFile = testHamPath + "\\" + testHamFile
        toOpenTestHamFile = open(openTestHamFile)
        allTestFiles.add(testHamFile)
        testHamFileSet.add(testHamFile)
        hamCount += 1.0
        presentFileWordCountMap = {}
        for line in toOpenTestHamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if not (useStopWords == 'no' and (stripped in stopWords)):
                        if stripped in presentFileWordCountMap:
                            presentFileWordCountMap[stripped] += 1.0
                        else:
                            presentFileWordCountMap[stripped] = 1.0
        testFileWordCountMap[testHamFile] = presentFileWordCountMap
        
        
    for testSpamFile in os.listdir(testSpamPath):
        openTestSpamFile = testSpamPath + "\\" + testSpamFile
        toOpenTestSpamFile = open(openTestSpamFile)
        allTestFiles.add(testSpamFile)
        testSpamFileSet.add(testSpamFile)
        spamCount += 1.0
        presentFileWordCountMap = {}
        for line in toOpenTestSpamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if not (useStopWords == 'no' and (stripped in stopWords)):
                        if stripped in presentFileWordCountMap:
                            presentFileWordCountMap[stripped] += 1.0
                        else:
                            presentFileWordCountMap[stripped] = 1.0
        testFileWordCountMap[testSpamFile] = presentFileWordCountMap    
    
    
    for file in allTestFiles:
        weightOfFile = weightOfTestFile(file)
        count = count+1
        if weightOfFile > 0.0:
            if file in testHamFileSet:
                hamMistake = hamMistake + 1
                mistake = mistake+1
        else:
            if file in testSpamFileSet:
                spamMistake = spamMistake+1
                mistake = mistake + 1
    
    print("Total emails",count)
    print("Total ham emails",hamCount)
    print("Total ham misclassifications",hamMistake)
    print("Total spam emails",spamCount)
    print("Total spam misclassifications",spamMistake)      
    print("Overall accuracy",(count-mistake)/count)    


def weightOfTestFile(filename):
    weightsum = w0
    presentWordMap = testFileWordCountMap[filename]
    for word in presentWordMap:
        if word in weightsMap:
            weightsum = weightsum+ (weightsMap[word]*presentWordMap[word])
    return weightsum
                
def weightOfTheFile(filename):
        weightsum = w0
        presentWordMap = fileWordCountMap[filename]
        for word in presentWordMap:
            weightsum = weightsum + (weightsMap[word]*presentWordMap[word])
        return sigmoid(round(weightsum,2))

def sigmoid(weightsum):
    exponential = 0.0
    try:
        exponential = math.exp(weightsum)
    except OverflowError:
        exponential = 5
    
    return exponential/(1+exponential)

def fillFileWeightsMap():
    for file in allFilesSet:
        weight = weightOfTheFile(file)
        fileWeightsMap[file] = weight

def updateWeights(newWeightsMap):
    for word in newWeightsMap:
        weightsMap[word] = newWeightsMap[word]

def train():
    for i in range(0,numberOfIterations):
        for word in listOfWords:
            error = 0.0
            for filename in allFilesSet:
                whichFile = 1.0
                presentWordMap = fileWordCountMap[filename]
                if filename in hamFileSet:
                    whichFile = 0.0
                weight = fileWeightsMap[filename]  
                if word in presentWordMap:
                    error = error + (presentWordMap[word])*(whichFile-weight)
            newWeight = weightsMap[word] + (learningRate*error) - (learningRate*lamda*weightsMap[word])
            newWeightsMap[word] = newWeight
        updateWeights(newWeightsMap)
        fillFileWeightsMap()

if __name__ == '__main__':
    main()
