import os
import glob
import csv

allFiles = ["" + x for x in  glob.glob("*.wav")];
trainFiles = []

with open("file2labels.csv", newline='') as csvfile:
    file2labelsReader = csv.reader(csvfile);
    for r in file2labelsReader:
        f = r[0] + ".wav"
        trainFiles.append(f)

#print(allFiles)
a = len(allFiles)
print(trainFiles)
b = len(trainFiles)

for train in trainFiles:
    print("Train : ", train)
    allFiles.remove(train)
    print("Count : ", train, " : ", trainFiles.count(train))

c = len(allFiles)

print("All files: ", a)
print("Train files: ", b)
print("Test files: ", c)


for f in allFiles:
    print("rename: ", f, " to ", os.path.join("test", f))
    os.rename(f, os.path.join("test", f))
