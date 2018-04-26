import os
import re
import math
import sys
def main():
    hamPath = os.getcwd() + "\\train\ham"
    spamPath = os.getcwd() + "\\train\spam"
    testHamPath = os.getcwd() + "\\test\ham"
    testSpamPath = os.getcwd() + "\\test\spam"
    listOfWords = []
    setWords = set()
    wordsMap = {}
    spamWordsMap = {}
    hamWordsMap = {}
    hamWordsLength = 0.0
    spamWordsLength = 0.0
    totalWordsLength = 0.0
    
    spamProbabilities = {}
    hamProbabilities = {}
    
    hams = 0.0
    spams = 0.0
    totalEmails = 0.0
    
    count = 0.0
    hamMistake = 0.0
    spamMistake = 0.0
    useStopWords = sys.argv[1]
    stopWords = set()
    
    if useStopWords == 'no':
        toOpenStopWordsFile = open(os.getcwd()+"\\stopwords.txt")
        for line in toOpenStopWordsFile:
            word = line.strip("\n")
            stopWords.add(word)
    
    
    for hamFile in os.listdir(hamPath):
        hams += 1.0
        totalEmails += 1.0
        openHamFile = hamPath + "\\" + hamFile
        toOpenHamFile = open(openHamFile)
        for line in toOpenHamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if not (useStopWords == 'no' and stripped in stopWords):
                        totalWordsLength += 1.0
                        hamWordsLength += 1.0
                        if stripped in wordsMap:
                            wordsMap[stripped] += 1.0
                            hamWordsMap[stripped] += 1.0
                        else:
                            wordsMap[stripped] = 1.0
                            hamWordsMap[stripped] = 1.0
                        
                        if not (stripped in setWords):
                            setWords.add(stripped)
                            listOfWords.append(stripped)
    for spamFile in os.listdir(spamPath):
        spams += 1.0
        totalEmails += 1.0
        openSpamFile = spamPath + "\\" + spamFile
        toOpenSpamFile = open(openSpamFile,encoding = 'latin-1')
        for spamLine in toOpenSpamFile:
            for spamWord in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', spamLine):
                for strippedSpamWord in spamWord.split():
                    if not (useStopWords == 'no' and strippedSpamWord in stopWords):
                        totalWordsLength += 1.0
                        spamWordsLength += 1.0
                        if strippedSpamWord in wordsMap:
                            wordsMap[strippedSpamWord] += 1.0
                            if strippedSpamWord in spamWordsMap:
                                spamWordsMap[strippedSpamWord] += 1.0
                            else:
                                spamWordsMap[strippedSpamWord] = 1.0
                        else:
                            wordsMap[strippedSpamWord] = 1.0
                            spamWordsMap[strippedSpamWord] = 1.0
                        if not (strippedSpamWord in setWords):
                            setWords.add(strippedSpamWord)
                            listOfWords.append(strippedSpamWord)
    
    hamProbability = hams/totalEmails
    spamProbability = spams/totalEmails
    
    for word in wordsMap:
        if word in hamWordsMap:
            hamProbabilities[word] = (hamWordsMap[word] + 1)/(hamWordsLength+len(wordsMap.keys()))
        else:
            hamProbabilities[word] = 1/(hamWordsLength+len(wordsMap.keys()))
        if word in spamWordsMap:
            spamProbabilities[word] = (spamWordsMap[word] + 1)/(spamWordsLength+len(wordsMap.keys()))
        else:
            spamProbabilities[word] = 1/(spamWordsLength+len(wordsMap.keys()))
    
    
    countHam = 0.0
    countSpam = 0.0
    
    for hamTestFile in os.listdir(testHamPath):
        count += 1.0
        countHam += 1.0
        hamLikelihood = 1.0
        spamLikelihood = 1.0
        hamPosterior = 0.0
        spamPosterior = 0.0
        openTestHamFile = testHamPath + "\\" + hamTestFile
        toOpenTestHamFile = open(openTestHamFile)
        for line in toOpenTestHamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if stripped in hamProbabilities or stripped in spamProbabilities:
                        hamLikelihood = hamLikelihood * (math.log(hamProbabilities[stripped]))
                        spamLikelihood = spamLikelihood*(math.log(spamProbabilities[stripped]))    
                        
        hamPosterior = hamProbability*hamLikelihood
        spamPosterior = spamProbability*spamLikelihood
        if hamPosterior < spamPosterior:
            hamMistake += 1.0
        
        
    for spamTestFile in os.listdir(testSpamPath):
        count += 1.0
        countSpam += 1.0
        hamLikelihood = 1.0
        spamLikelihood = 1.0
        hamPosterior = 0.0
        spamPosterior = 0.0
        openTestSpamFile = testSpamPath + "\\" + spamTestFile
        toOpenTestSpamFile = open(openTestSpamFile)
        for line in toOpenTestSpamFile:
            for word in re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', line):                
                for stripped in word.split():
                    if stripped in hamProbabilities or stripped in spamProbabilities:
                        hamLikelihood = hamLikelihood * (math.log(hamProbabilities[stripped]))
                        spamLikelihood = spamLikelihood * (math.log(spamProbabilities[stripped]))    
                    
        hamPosterior = hamProbability*hamLikelihood
        spamPosterior = spamProbability*spamLikelihood    
        if hamPosterior > spamPosterior:
            spamMistake += 1.0
    accuracy = (count-(hamMistake+spamMistake))/count    
    print('total count of emails',count)
    print("Total Ham emails",countHam)
    print("Total ham misclassifications",hamMistake)
    print("Total spam emails",countSpam)
    print("Total spam misclassifications",spamMistake)  
    print("Overall accuracy is " , accuracy) 
        
if __name__ == '__main__':
    main()