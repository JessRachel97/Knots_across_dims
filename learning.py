import numpy as np
from tensorflow import keras
import ast
import csv
from sklearn.model_selection import train_test_split
import time

train_size = 0.25  # between 0 and 1, the fraction of data used for training
epochs = 50 # number of passes through the dataset
density = 100 # nodes per layer in the neural network
activation = 'relu' # activtion function
loss = 'sparse_categorical_crossentropy' # loss function
optim = 'adam' # optimizer


# trains a neural network to learn inputs from outputs.
# parameters are numpy arrays.
def learn(inputs, outputs):
    dims = len(inputs[0])
    num_classes = max(outputs) - min(outputs) + 1

    train_in, test_in, train_out, test_out = train_test_split(inputs, outputs,
                                                train_size=train_size)

    model = keras.models.Sequential([
            keras.layers.Dense(density, activation=activation, input_dim=dims),
            keras.layers.Dense(density, activation=activation),
            keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
    model.fit(train_in, train_out, epochs=epochs, verbose=1)
    return model, test_in, test_out

# determine which inputs and outputs to use
inn = ''
while inn not in ['1', '2'] :
    inn = input('Select input invariant. (1) Khovanov polynomial (2) Jones polynomial\n')
outt = ''
while outt not in ['1', '2']:
    outt = input('Select output invariant. (1) s invariant (2) slice genus\n')

ins = []
outs = []
max_len = 0

# read in data from files and prepare for learning
knot_file = open('knot_table_july25_with_KH_width.csv')
reader = csv.reader(knot_file, delimiter=',')
data_list = list(reader)
knot_file.close()
titles = data_list[0]
del(data_list[0])
if inn == '1':
    in_idx = titles.index('Khovanov Polynomial')
else:
    in_idx = titles.index('Jones Polynomial')
if outt == '1':
    out_idx = titles.index('Rasmussen s-invariant')
else:
    out_idx = titles.index("Slice genus lower bound")
for l in data_list:
    try:
        in_curr = ast.literal_eval(l[in_idx])
        out_curr = int(l[out_idx])
        if outt == '2':
            out_curr_2 = int(l[out_idx + 1])
        in_curr = [item for sublist in in_curr for item in sublist]
        if len(in_curr) > max_len:
            max_len = len(in_curr)
        if outt == '2' and out_curr == out_curr_2:
                ins.append(in_curr)
                outs.append(out_curr)
        else:
                ins.append(in_curr)
                outs.append(out_curr)
    except:
        pass


if outt == '1':
    knot_file = open('independentRandomPDCodeTable2.csv')
    reader = csv.reader(knot_file, delimiter=',')
    data_list = list(reader)
    knot_file.close()
    titles = data_list[0]
    del(data_list[0])
    if inn == '1':
        in_idx = titles.index('Khovanov_polynomial')
    else:
        in_idx = titles.index('Jones_polynomial')
    out_idx = titles.index('s_invariant')
    for l in data_list:
        try:
            in_curr = ast.literal_eval(l[in_idx])
            out_curr = int(l[out_idx])
            in_curr = [item for sublist in in_curr for item in sublist]
            if len(in_curr) > max_len:
                max_len = len(in_curr)
            ins.append(in_curr)
            outs.append(out_curr)
        except:
            pass

if outt == '1':
    knot_file = open('KnotTableNov16.csv')
    reader = csv.reader(knot_file, delimiter=',')
    data_list = list(reader)
    knot_file.close()
    titles = data_list[0]
    del(data_list[0])
    if inn == '1':
        in_idx = titles.index('Khovanov polynomial')
    else:
        in_idx = titles.index('Jones polynomial')
    out_idx = titles.index('Rasmussen s-invariant')
    for l in data_list:
        try:
            in_curr = ast.literal_eval(l[in_idx])
            out_curr = int(l[out_idx])
            in_curr = [item for sublist in in_curr for item in sublist]
            if len(in_curr) > max_len:
                max_len = len(in_curr)
            ins.append(in_curr)
            outs.append(out_curr)
        except:
            pass

# pad polynomials with zeros for uniform length
for i in range(len(ins)):
    pol = ins[i]
    while len(pol) < max_len:
        pol.append(0)
    ins[i] = pol

# rescale outputs so that minimum is zero
out_min = min(outs)
outs = [o - out_min for o in outs]

print(f'BEGINNING TRAINING. {len(ins)} KNOTS INCLUDED.\n')

ins = np.array(ins)
outs = np.array(outs)
model, test_in, test_out = learn(ins, outs)

# from here you can use model, test_in, test_out, to validate and test your ideas
