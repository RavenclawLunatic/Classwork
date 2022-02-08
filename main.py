'''
BGDatasetAnalysis is a Python project that takes any amount of mechanics
and tells you if they've worked before in one of top 100 games on BGG
2/3/2022
author = CharlotteMiller
'''

import math
from pandas import read_csv
from pandas.plotting import scatter_matrix
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
url = "~/Desktop/Advanced CS/Datasets/bgg_dataset.csv"
names = ['ID', 'Name', 'YearPublished', 'MinPlayers', 'MaxPlayers',	'PlayTime', 'MinAge', 'UsersRated', 'RatingAverage', 'BGGRank', 'ComplexityAverage', 'OwnedUsers', 'Mechanics', 'Domains']
dataset = read_csv(url, names=names, delimiter=";")
# numpy made the data easier to work with
numpyset = dataset.to_numpy()
# both of the below are a list full of lists where each list of mechanics/domains
# has the same index as its corresponding game in the regular dataset (eg. all
# the Gloomhaven stuff is at index 1
mechanicsSet = []
domainsSet = []

for idx, item in enumerate(numpyset):
    if idx == 0:
        mechanicsSet.append([])
        domainsSet.append([])
        continue
    if pd.isna(item[12]):
        mechanicsSet.append([])
    else:
        mechanicsSplit = item[12].split(", ")
        mechanicsSet.append(mechanicsSplit)
    if pd.isna(item[13]):
        domainsSet.append([])
    else:
        domainsSplit = item[13].split(", ")
        domainsSet.append(domainsSplit)

def isGood(listOfMechanicsIdea):
    count = 0
    potential = []
    for ideaItem in listOfMechanicsIdea:
        print(ideaItem)
        i = 0
        while i <= 100:
            print("i:", i)
            j = 0
            while j < len(mechanicsSet[i]):
                print("j:", j)
                temp = mechanicsSet[i][j]
                if temp == ideaItem:
                    potential.append(numpyset[i][1])
                    print(potential)
                j += 1
            i += 1
    for item in potential:
        if potential.count(item) >= len(listOfMechanicsIdea):
            print(item)
            return True, item
    return False


mechanicsIdea = input("What mechanics do you want your game to have \n(format: \"Item One, Item Two, Item Three, etc.\"): ")
listOfMechanicsIdea = mechanicsIdea.split(', ')
bool, item = isGood(listOfMechanicsIdea)
if bool:
    print("Your idea of using", mechanicsIdea, "is one people have previously liked, such as in " + item + ", so go for it!")
else:
    print("Your idea of using", mechanicsIdea, "is one people haven't previously liked, so it may not be the best idea")

'''
BestMechanics = []
TallyOfBM = []
HowManyMechanics = []
for i in range(0,101):
    for item in dataset.loc[dataset.BGGRank == str(i), 'Mechanics']:
        thing = item.split(", ")
        HowManyMechanics.append(len(thing))
        for j in range(len(thing)):
            if thing[j] in BestMechanics:
                TallyOfBM[BestMechanics.index(thing[j])] += 1
            else:
                BestMechanics.append(thing[j])
                TallyOfBM.append(1)
print("These are all the mechanics found in the 100 best games according to Board Game Geek:", BestMechanics)
print("This shows how many of those games had a specific mechanic (eg. 6 of the top 100 games had Action Queue):", TallyOfBM)
print("This shows how many mechanics each of the top 100 games had (eg. Gloomhaven had 19 mechanics):", HowManyMechanics)
print("This is the average number of mechanics of the top 100 games:", sum(HowManyMechanics)/len(HowManyMechanics))
print()

BestDomains = []
TallyOfBD = []
HowManyDomains = []
for i in range(0,101):
    for item in dataset.loc[dataset.BGGRank == str(i), 'Domains']:
        thing = item.split(", ")
        HowManyDomains.append(len(thing))
        for j in range(len(thing)):
            if thing[j] in BestDomains:
                TallyOfBD[BestDomains.index(thing[j])] += 1
            else:
                BestDomains.append(thing[j])
                TallyOfBD.append(1)
print("These are all the domains found in the 100 best games according to Board Game Geek:", BestDomains)
print("This shows how many of those games had a specific domain (eg. 78 of the top 100 games had Strategy Games):", TallyOfBD)
print("This shows how many domains each of the top 100 games had (eg. Gloomhaven had 2 domains):", HowManyDomains)
print("This is the average number of domains of the top 100 games:", sum(HowManyDomains)/len(HowManyDomains))
print()

BestPlayerMin = []
TallyOfBPMi = []
HowManyPlayerMins = []
for i in range(0,101):
    for item in dataset.loc[dataset.BGGRank == str(i), 'MinPlayers']:
        thing = item.split(", ")
        HowManyPlayerMins.append(int(item))
        for j in range(len(thing)):
            if thing[j] in BestPlayerMin:
                TallyOfBPMi[BestPlayerMin.index(thing[j])] += 1
            else:
                BestPlayerMin.append(thing[j])
                TallyOfBPMi.append(1)
print("These are the minimum amount of players found in the 100 best games according to Board Game Geek:", BestPlayerMin)
print("This shows how many of those games had a specific player minimum (eg. 42 of the top 100 games had 1 player as their minimum):", TallyOfBPMi)
print("This is the average number of minimum players:", sum(HowManyPlayerMins)/len(HowManyPlayerMins))
print()

BestPlayerMax = []
TallyOfBPMa = []
HowManyPlayerMaxs = []
for i in range(0,101):
    for item in dataset.loc[dataset.BGGRank == str(i), 'MaxPlayers']:
        thing = item.split(", ")
        HowManyPlayerMaxs.append(int(item))
        for j in range(len(thing)):
            if thing[j] in BestPlayerMax:
                TallyOfBPMa[BestPlayerMax.index(thing[j])] += 1
            else:
                BestPlayerMax.append(thing[j])
                TallyOfBPMa.append(1)
print("These are the maximum amount of players found in the 100 best games according to Board Game Geek:", BestPlayerMax)
print("This shows how many of those games had a specific player maximum (eg. 59 of the top 100 games had 4 players as their maximum):", TallyOfBPMa)
print("This is the average number of maximum players:", sum(HowManyPlayerMaxs)/len(HowManyPlayerMaxs))
print()
'''

"""
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""