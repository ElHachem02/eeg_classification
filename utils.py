import pandas as pd

# Example usage
data_folder = '/Users/peter/Documents/Academy/University/ETH/courses/ma_2/ai_project/EEG-Transformer/data/experiment/grouped_data/x_points'  # replace with your data folder path
labels_folder = '/Users/peter/Documents/Academy/University/ETH/courses/ma_2/ai_project/EEG-Transformer/data/experiment/grouped_data/labels'  # replace with your labels folder path
subjects = ['lea','finn','sarah', 'aurora', 'bjoern', 'derek', 'dimi', 'ronan'] # add all subjects here
# ,'finn','sarah', 'aurora', 'bjoern', 'derek', 'dimi', 'ronan'

SAMPLING_RATE = 256
SEED = 42

def _to_true_label(label):
    if label == 100:
        raise Exception("Must skip labels with value 100!")
    if label == 195:
        return 1
    if label == 196:
        return 2
    return 0

def _split_data(data, labels, timestamps):    
    x = []
    y = []
    start = timestamps[0]
    for i, label in enumerate(labels):
        if i == 0: continue
        end = timestamps[i]
        if label != 100:
            x.append(data[int(start):int(end)])
            y.append(_to_true_label(label))
        start = timestamps[i]
    y = pd.Series(y)
    return (x,y)

# Data will be of the form
'''
Ronan -> ([list of df], [list of labels])
'''

def _generate_subject_data():
    subj_data = {}
    for subj in subjects:
        print(subj)
        df = pd.read_csv(labels_folder+"/events_" + subj + ".txt", delim_whitespace=True)
        df = df[(df.number != "condition")]
        subj_data[subj] = {}
        subj_data[subj]["labels"] = df["number"].to_numpy().astype(float)
        subj_data[subj]["timestamps"] = df["type"].to_numpy().astype(float)
        if subj == 'aurora': # aurora is another format
            df = pd.read_csv(data_folder+"/" + subj + "_pre_processed_data.txt", delim_whitespace=True)
        else:
            df = pd.read_csv(data_folder+"/" + subj + "_pre_processed_data.txt", delim_whitespace=False)
        subj_data[subj]["data"] = df
                
    for x in subjects:
        if subj_data[x]['labels'][0] != 100:
            print ("Something wrong with labels for " + x)
    
    return subj_data

def load_data_and_labels():
    subj_data = _generate_subject_data()
    processed_subjects = {}
    for s in subjects:
        processed_subjects[s] = _split_data(subj_data[s]['data'], subj_data[s]['labels'], subj_data[s]['timestamps'])
        
    return processed_subjects

def _split_texts_LTO(data, labels, timestamps):
    texts = []
    x = []
    y = []
    start = timestamps[0]
    for i, label in enumerate(labels):
        if i == 0: continue
        end = timestamps[i]
        if label != 100:
            x.append(data[int(start):int(end)])
            y.append(_to_true_label(label))
        else:
            texts.append((x,y))
            x = []
            y = []
        start = timestamps[i]
    texts.append((x,y))
    if len(texts) != 3:
        raise Exception("Texts must be 3, not " + str(len(texts)))
    return texts

def load_data_LTO():
    subj_data = _generate_subject_data()
    X1 = []
    X2 = []
    X3 = []
    y1 = []
    y2 = []
    y3 = []
    for subj in subjects:
        print(subj)
        texts = _split_texts_LTO()(subj_data[subj]['data'], subj_data[subj]['labels'], subj_data[subj]['timestamps'])
        X1 += texts[0][0]
        y1 += texts[0][1]
        X2 += texts[1][0]
        y2 += texts[1][1]
        X3 += texts[2][0]
        y3 += texts[2][1]
