import csv

def test():
    with open("CVfolds_2.txt", newline='') as id2set, open("rec_id2filename.txt", newline='') as id2file, open("rec_labels_test_hidden.txt", newline='') as id2label:
        with open("file2label.csv", 'w', newline='') as file2label:
            readId2Label = csv.reader(id2label)
            readId2Set = csv.reader(id2set)
            readId2File = csv.reader(id2file)
            file2labelwriter = csv.writer(file2label)

            id2file = {}
            for r in readId2File:
                if r[0] == 'rec_id':
                    print("Reading id to file...")
                else:
                    id2file[r[0]] = r[1]

            print("Done reading id to file.")

            nb_samples = 0
            nb_bird_present = 0

            print("Creating file to labels csv...")
            for (id2label, id2set) in zip(readId2Label, readId2Set):
                if(id2set[0] != id2label[0]):
                    raise ValueError
                iden = id2set[0]
                if(id2set[1] == '0'):
                    nb_samples += 1
                    if(len(id2label) > 1):
                        labels = id2label[1:]
                        nb_bird_present += 1
                        f = id2file[iden]
                        file2labelwriter.writerow([f] + labels)
                    else:
                        file2labelwriter.writerow([f])

            print("Number of training samples: ", nb_samples)
            print("Number of training samples with birds present: ", nb_bird_present)
