import csv
from tp1 import Neuralnetworks

# Read tp1_data
digit_data = []
classes = []

with open('data/data_tp1') as csvfile:
  digits  = csv.reader(csvfile)
  for row in digits:
    row_numeric = [int(numeric_string) for numeric_string in row]
    #print("class = " + row[0])
    classes.append(row_numeric[0])
    digit_data.append(row_numeric[1:])

a = Neuralnetworks(784,10,10)


#a = Neuralnetworks(2,1,2)
#a.train(1,[[1,1],[2,2],[3,3],[4,4]],[1,1,1,1],1,4)
a.train(1,digit_data,classes,1,10)
