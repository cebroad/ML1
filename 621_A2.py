import csv
import sys
import numpy as np
import HTMLParser
from string import lowercase, uppercase
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import time

def gettweets(filename):
    tweets = []
    classes = []
    with open(filename, 'rb') as file:
        filereader = csv.reader(file)
        for line in filereader:
            classes.append(line[0])
            tweet = line[1:]            # in order to put the chunks of the tweet back together
            tweet = ','.join(tweet)
            tweet = tweet[2:-1]         # remove extra double quotes + space from beginning and end
            tweets.append(tweet)
    return tweets, classes

def salad(tweet):
    tweet = unicode(tweet, encoding='latin-1')
    h = HTMLParser.HTMLParser()
    tweet = h.unescape(tweet)           # convert HTML objects
    tweet = tweet.lower()
    tokens = []
    current = ''
    letters = lowercase + "'" + '0123456789' + '@'
    for char in tweet:
        if char in letters:
            current = current + char
        else:
            tokens.append(current)
            tokens.append(char)
            current = ''
    final = []
    for item in tokens:
        if item != '':
            if item != ' ':
                final.append(item)
    return final

def saladAC(tweet):
    tweet = unicode(tweet, encoding='latin-1')
    h = HTMLParser.HTMLParser()
    tweet = h.unescape(tweet)  # convert HTML objects
    tokens = []
    current = ''
    letters = lowercase + uppercase + "'" + '0123456789' + '@' + '#'
    for char in tweet:
        if char in letters:
            current = current + char
        else:
            tokens.append(current)
            tokens.append(char)
            current = ''
    final = []
    for item in tokens:
        if item != '':
            if item != ' ':
                final.append(item)
    return final

def repeatchar(words):
    count = 1
    overall = 0
    for word in words:
        for i in range(1, len(word) - 1):
            if word[i] == word[i-1]:
                if word[i+1] == word[i]:
                    count += 1
    if count > 0:
        overall = 1
    return count, overall

def repeatword(words):
    longwords = []
    for word in words:
        if len(word) > 1:
            longwords.append(word)
    wordcount = Counter(longwords)
    repeats = []
    repeatsTF = 0
    for word in longwords:
        if wordcount[word] > 1:
            repeats.append(word)
    if len(repeats) > 0:
        repeatsTF = 1
    return len(repeats), repeatsTF

def findAC(words):
    count = 0
    ACTF = 0
    for word in words:
        capscount = 0
        for letter in words:
            if letter in uppercase:
                capscount += 1
        if capscount == len(word):
            count += 1
    if count > 1:
        ACTF = 1
    return count, ACTF

def maxlen(words):
    maxlength = 0
    for word in words:
        if len(word) > maxlength:
            maxlength = len(word)
    return maxlength

def POS(words):
    POSs = []
    taglist = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    tagged = pos_tag(words)
    POScount = Counter()
    for word in tagged:
        POScount[word[1]] += 1
    for tag in taglist:
        try:
            POSs.append(POScount[tag]/float(len(words)))
        except:
            POSs.append(0)
    return POSs

def SW(words):
    count = 0
    for word in words:
        if word in stopwords.words('english'):
            count += 1
    return count

def retweet(words):
    RTTF = 0
    for word in words:
        if word == 'rt':
            RTTF += 1
    if RTTF > 0:
        RTTF = 1
    return RTTF

def lengthmap(words):
    lm = Counter()
    for word in words:
        length = len(word)
        if length < 9:
            lm[length] += 1
        else:
            lm[9] += 1
    ratios = []
    for i in range(1,10):
        ratio = lm[i]/float(len(words))
        ratios.append(ratio)
    return ratios

def getfeatures(tweets, other):
    features = []
    testset = ["i", "the", "and", "to", "a", "of", "that", "in", "it", "my", "is", "you", "was", "for", "have", "with", "he", "me", "on", "but", ".", ",", "!"]
    happy = [':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'] #from http://www.nltk.org/_modules/nltk/sentiment/util.html
    sad = [':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('] #same source
    for word in other:
        if word not in testset:
            testset.append(word)
    #print len(testset)
    #for face in happy:
    #    if face not in testset:
    #        testset.append(face)
    #for face in sad:
    #    if face not in testset:
    #        testset.append(face)
    for i in range(len(tweets)):
        tokens = salad(tweets[i])
        num = len(tokens)
        counts = Counter(tokens)
        tweetfeat = []
        for j in range(len(testset)):
            rate = counts[testset[j]]/float(num)
            tweetfeat.append(rate)
        tweetfeat.append(num)
        atcount = 0
        for word in tokens:
            if '@' in word:
                atcount += 1
        tweetfeat.append(atcount/float(num))
        hashcount = 0
        for word in tokens:
            if '#' in word:
                hashcount += 1
        tweetfeat.append(hashcount/float(num))
        #tweetfeat.append(hashcount)
        notcount = 0
        for word in tokens:
            if "not" or "n't" in word:
                notcount += 1
        tweetfeat.append(float(notcount)/num)
        repeatc, repeatcTF = repeatchar(tokens)
        #tweetfeat.append(repeatc)
        tweetfeat.append(repeatcTF)
        repeatw, repeatwTF = repeatword(tokens)
        #tweetfeat.append(repeatw)
        tweetfeat.append(repeatwTF)
        tokensAC = saladAC(tweets[i])
        allcaps, allcapsTF = findAC(tokensAC)
        #tweetfeat.append(allcaps)
        tweetfeat.append(allcapsTF)
        maxlength = maxlen(tokens)
        tweetfeat.append(maxlength)
        POScounts = POS(tokens)
        for count in POScounts:
            tweetfeat.append(count)
        stops = SW(tokens)
        tweetfeat.append(stops)
        RT = retweet(tokens)
        tweetfeat.append(RT)
        happyface = 0
        sadface = 0
        for face in happy:
            if face in tweets[i]:
                happyface += 1
        for face in sad:
            if face in tweets[i]:
                sadface += 1
        tweetfeat.append(happyface)
        tweetfeat.append(sadface)
        lenmap = lengthmap(tokens)
        for ratio in lenmap:
            tweetfeat.append(ratio)
        features.append(tweetfeat)
    featarray = np.array(features)
    return featarray

def checkpred(obs, pred):
    num = len(obs)
    true = 0
    for i in range(num):
        if obs[i] == pred[i]:
            true += 1
    return float(true)/num

def knn(train_X, train_Y, test_X, test_Y):
    bestk = 0
    bestrate = 0.0
    for i in range(1,20):
        knnclass = KNeighborsClassifier(n_neighbors = i)
        knnclass.fit(train_X, train_Y)
        pred = knnclass.predict(test_X)
        rate = checkpred(test_Y, pred)
        if rate > bestrate:
            bestk = i
            bestrate = rate
    return (1 - bestrate), bestk

def logistic(train_X, train_Y, test_X, test_Y):
    logreg = LogisticRegression()
    logreg.fit(train_X, train_Y)
    pred = logreg.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate)

def lindesc(train_X, train_Y, test_X, test_Y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_Y)
    pred = lda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate)

def quadpred(train_X, train_Y, test_X, test_Y):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_X, train_Y)
    pred = qda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate)

def main(wordlist):
    try:
        train = sys.argv[1]
    except:
        print 'Sorry, training set is a required input'
        exit()
    try:
        test = sys.argv[2]
    except:
        print 'Sorry, test set is a required input'
        exit()
    train_tweets, train_classes  = gettweets(train)
    test_tweets, test_classes = gettweets(test)
    train_feat = getfeatures(train_tweets, wordlist)
    test_feat = getfeatures(test_tweets, wordlist)
    knnpred, k = knn(train_feat, train_classes, test_feat, test_classes)
    #print 'Missclassification for K Nearest Neighbors with ' + str(k) + ' k neighbors is ' + str(knnpred)
    logregpred = logistic(train_feat, train_classes, test_feat, test_classes)
    #print 'Missclassification for Logistic Regression is ' + str(logregpred)
    LDApred = lindesc(train_feat, train_classes, test_feat, test_classes)
    #print 'Missclassification for LDA is ' + str(LDApred)
    QDApred = quadpred(train_feat, train_classes, test_feat, test_classes)
    #print 'Missclassification for QDA is ' + str(QDApred)
    bestpred = min([knnpred, logregpred, LDApred, QDApred])
    print 'Misclassification rate: ' + str(bestpred)

train_tweets, train_classes = gettweets(sys.argv[1])

positivewords = Counter()
negativewords = Counter()
neutralwords = Counter()

for i in range(len(train_tweets)):
    tokens = salad(train_tweets[i])
    if train_classes[i] == 'positive':
        for word in tokens:
            positivewords[word] += 1
    elif train_classes[i] == 'negative':
        for word in tokens:
            negativewords[word] += 1
    elif train_classes[i] == 'neutral':
        for word in tokens:
            neutralwords[word] += 1
    else:
        pass

toppos = positivewords.most_common(150)
topneg = negativewords.most_common(150)
topneut = neutralwords.most_common(150)

poslist = []
neglist = []
neutlist = []

for word in toppos:
    poslist.append(word[0])

for word in topneg:
    neglist.append(word[0])

for word in topneut:
    neutlist.append(word[0])

important = []

for word in poslist:
    important.append(word)
for word in neglist:
    if word not in important:
        important.append(word)
for word in neutlist:
    if word not in important:
        important.append(word)

'''
count = 0
for word in positivewords:
    if word not in negativewords:
        if word not in important:
            lettercount = 0
            for letter in word:
                if letter in lowercase + uppercase:
                    lettercount += 1
            if lettercount == len(word):
                important.append(word)
                count += 1
                print word
                if count > 20:
                    break

count = 0
for word in negativewords:
    if word not in positivewords:
        if word not in important:
            lettercount = 0
            for letter in word:
                if letter in lowercase + uppercase:
                    lettercount += 1
            if lettercount == len(word):
                important.append(word)
                count += 1
                print word
                if count > 20:
                    break
'''
# print len(important)

start_time = time.time()

main(important)

end_time = time.time()

#  "time elapsed: " + str(end_time - start_time)