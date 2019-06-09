
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import json


# In[3]:

with open('cid_cid_to_feature_vectors.json', 'r') as f:
    cid_cid_to_feature_vectors = json.load(f)
with open('cid_cid_to_rating_combined.json', 'r') as f:
    cid_cid_to_rating = json.load(f)
with open('side_effect_code_to_index.json', 'r') as f:
    side_effect_code_to_index = json.load(f)
with open('side_effect_code_to_name.json', 'r') as f:
    side_effect_code_to_name = json.load(f)
# get codes sorted by index
side_effect_codes = [kv[0] for kv in sorted(side_effect_code_to_index.items(), key=lambda kv: kv[1])]
side_effect_names = [side_effect_code_to_name[code] for code in side_effect_codes]
feature_vectors_df=pd.DataFrame.from_dict(cid_cid_to_feature_vectors, orient='index')
feature_vectors_df.columns = side_effect_names
feature_vectors_df['cid_cid'] = feature_vectors_df.index
labels_df=pd.DataFrame.from_dict(cid_cid_to_rating, orient='index')
labels_df.columns = ["severity"]
labels_df['cid_cid'] = labels_df.index


# In[4]:

data_df = pd.merge(feature_vectors_df, labels_df, how='inner', on=['cid_cid'])
data_df.to_csv("lexi-data_combined.csv",index=False, header=True)
data_df.head(10)


# In[12]:

# load data
data_df = pd.read_csv("lexi-data_combined.csv")
data_df.head(10)


# In[5]:

# baseline data collection - pick a few columns we think are representative
with open('side_effect_code_to_index.json', 'r') as f:
    side_effect_code_to_index = json.load(f)
columns = ["Difficulty breathing", "narcolepsy", "diarrhea", "AFIB", "emesis"]
codes = ["C0020672", "C0014863", "C0004096", "C0032768", "C0034065"]
indicies = [side_effect_code_to_index[code] for code in codes]
baseline_data_df = data_df[["Difficulty breathing", "narcolepsy", "diarrhea", "AFIB", "emesis"]]
baseline_labels_df =  data_df[["severity"]]


# In[6]:

# baseline prediction - simple multinomial logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data_train, data_test, labels_train, labels_test = train_test_split(baseline_data_df.values, baseline_labels_df.values, test_size=0.2)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(data_train, labels_train)
preds = clf.predict(data_test)
accuracy_score(labels_test, preds) 


# In[7]:

# just a majority classifier
numA = (labels_train == 'A').sum()
numB = (labels_train == 'B').sum()
numC = (labels_train == 'C').sum()
numD = (labels_train == 'D').sum()
numX = (labels_train == 'X').sum()


# In[64]:

# let's try using SVD (like sparse PCA) to cut down on our feature vector length
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
svd = TruncatedSVD(n_components=200, n_iter=25)
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
monitor_df = data_df[data_df.severity=='C']
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=True, n_samples=2065)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
labels_df = df_upsampled[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"

trunc_all_values = svd.fit_transform(upsampled_values)
data_train, data_test, labels_train, labels_test = train_test_split(trunc_all_values, labels_df.values, test_size=0.2)


# In[23]:

# let's get all of the features
from sklearn.model_selection import train_test_split
labels_df = data_df[["severity"]]
all_data = data_df.drop(["severity", "cid_cid"], axis=1)
all_values = all_data.values
data_train, data_test, labels_train, labels_test = train_test_split(all_values, labels_df.values, test_size=0.2)


# In[29]:

# let's try and simplify the problem into 3 classes: 1, 2, 3
# A/B -> 1
# C -> 2
# D/X -> 3
labels_df = data_df[["severity"]]
from sklearn.model_selection import train_test_split
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
data_train, data_test, labels_train, labels_test = train_test_split(all_values, labels_df.values.flatten(), test_size=0.2)


# In[95]:

# let's try minority class up-sampling
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
monitor_df = data_df[data_df.severity=='C']
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=True, n_samples=2065)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
labels_df = df_upsampled[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
data_train, data_test, labels_train, labels_test = train_test_split(upsampled_values, labels_df.values.flatten(), test_size=0.2)


# In[37]:

# let's try majority class downsampling
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
monitor_df = data_df[data_df.severity=='C']
monitor_df = resample(monitor_df, replace=False, n_samples=169) 
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=False, n_samples=169)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
labels_df = df_upsampled[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
data_train, data_test, labels_train, labels_test = train_test_split(upsampled_values, labels_df.values.flatten(), test_size=0.2)


# In[138]:

# let's try thresholding w/ minority class up-sampling
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
monitor_df = data_df[data_df.severity=='C']
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=True, n_samples=2065)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
labels_df = df_upsampled[["severity"]]
df_upsampled = df_upsampled.drop(["severity", "cid_cid"], axis=1)
# get list of columns to keep
threshold = 400
columns_to_keep = []
for col, count in side_effect_counts.items():
    if count > threshold:
        columns_to_keep.append(col)
df_upsampled = df_upsampled[columns_to_keep]
upsampled_values = df_upsampled.values
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
data_train, data_test, labels_train, labels_test = train_test_split(upsampled_values, labels_df.values.flatten(), test_size=0.2)


# In[139]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(data_train, labels_train)
# clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight={"1":2, "2":1, "3":2}).fit(data_train, labels_train)
preds = clf.predict(data_test)
print(accuracy_score(labels_test, preds))
train_preds = clf.predict(data_train)
print("train accuracy: " + str(accuracy_score(labels_train, train_preds)))


# In[120]:

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#clf = SVC(gamma='scale', decision_function_shape='ovo', class_weight={"A":3, "B":4, "C":1, "D": 4, "X":5})
# clf = SVC(gamma='scale', decision_function_shape='ovo', class_weight={"1":2, "2":1, "3":2})
clf = SVC(decision_function_shape='ovo', kernel='rbf', C=100000, gamma=10)

clf.fit(data_train, labels_train)
preds = clf.predict(data_test)
accuracy_score(labels_test, preds) 
print(accuracy_score(labels_test, preds))
train_preds = clf.predict(data_train)
print("train accuracy: " + str(accuracy_score(labels_train, train_preds)))
print(classification_report(labels_test, preds))


# In[65]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
# clf = RandomForestClassifier(class_weight="balanced")
# clf = RandomForestClassifier(class_weight={"1":2, "2":1, "3":2})
clf.fit(data_train, labels_train)
preds = clf.predict(data_test)
accuracy_score(labels_test, preds)
print(accuracy_score(labels_test, preds))
train_preds = clf.predict(data_train)
print("train accuracy: " + str(accuracy_score(labels_train, train_preds)))


# In[45]:

preds


# In[96]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(data_train, labels_train)
preds = clf.predict(data_test)
accuracy_score(labels_test, preds) 
print(accuracy_score(labels_test, preds))
train_preds = clf.predict(data_train)
print("train accuracy: " + str(accuracy_score(labels_train, train_preds)))


# In[ ]:

# recursive feature elimination 
from sklearn.feature_selection import RFE
estimator = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=350)
# num features to select 1500, 1000, 500
selector = RFE(estimator, 1000)
selector = selector.fit(data_train, labels_train)
preds = selector.predict(data_test)
print(accuracy_score(labels_test, preds))
train_preds = selector.predict(data_train)
print("train accuracy: " + str(accuracy_score(labels_train, train_preds)))


# In[102]:

side_effect_counts = {}
counts_df = data_df.drop(["severity", "cid_cid"], axis=1).sum(axis = 0, skipna = True)

for index, row in counts_df.iteritems():
    side_effect_counts[index] = row


# In[125]:

import matplotlib.pyplot as plt
side_effects = []
counts = []
for effect, count in side_effect_counts.items():
    side_effects.append(effect)
    counts.append(count)
y_pos = np.arange(len(side_effects))
plt.bar(y_pos, counts, color = (0.5,0.1,0.5,0.6))
plt.title('Distribution of side effect counts')
plt.xlabel('side effect')
plt.ylabel('count')
plt.xticks(y_pos, counts)
plt.show()


# In[121]:

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm

matrix = confusion_matrix(labels_test, preds)
classes = unique_labels(labels_test, preds)
matrix  = matrix .astype('float') / matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(matrix , interpolation='nearest', cmap=cm.RdPu)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(matrix.shape[1]),
       yticks=np.arange(matrix.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title="SVM DDI Severity Classification: Confusion Matrix",
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = matrix.max() / 2.
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, format(matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black")
fig.tight_layout()
plt.show()


# In[161]:

indexes = []
for i in range(len(preds)):
    if preds[i] != labels_test[i] and preds[i] == '1' and labels_test[i]=='2':
        indexes.append(i)


# In[162]:

indexes # talk about systemic v. topical


# In[ ]:

# let's visualize our data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D


labels_df = data_df[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
df = data_df.drop(["severity", "cid_cid"], axis=1)
pca = PCA(n_components=3)
pca.fit(df)
result=pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)
colors = ['red','green','blue', 'purple']

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=labels_df.values, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()


# In[92]:

# let's try using SVD (like sparse PCA) to cut down on our feature vector length
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def run_pca(num_components, clf, upsampled_values, labels_df):
    svd = TruncatedSVD(n_components=num_components, n_iter=25)
    trunc_all_values = svd.fit_transform(upsampled_values)
    min_val = min(np.ravel(trunc_all_values))
    trunc_all_values += -1*min_val
    data_train, data_test, labels_train, labels_test = train_test_split(trunc_all_values, np.ravel(labels_df.values), test_size=0.2)
    clf.fit(data_train, labels_train)
    preds = clf.predict(data_test)
    test_acc = accuracy_score(labels_test, preds)
    train_preds = clf.predict(data_train)
    train_acc = accuracy_score(labels_train, train_preds)
    return test_acc, train_acc
    
    


# In[129]:

# determine best pca
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
monitor_df = data_df[data_df.severity=='C']
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=True, n_samples=2065)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
labels_df = df_upsampled[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"

train = []
test = []
num_comp = []
# for num in range(1000,100,-100):
for num in range(1, 2):
    # model = RandomForestClassifier(n_estimators=10)
    clf = MultinomialNB()
    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # model = SVC(gamma='scale', decision_function_shape='ovo', kernel='rbf')
    test_acc, train_acc = run_pca(700, model, upsampled_values, labels_df)
    train.append(train_acc)
    print(train_acc)
    test.append(test_acc)
    print(test_acc)
    num_comp.append(num)


# In[94]:

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(num_comp, train, label="train accuracy", color="purple")
plt.plot(num_comp, test, label="test accuracy", color="cyan")
plt.title("Naive Bayes PCA analysis")
plt.ylabel("Accuracy")
plt.xlabel("Number of components")
plt.hlines(y=.55, xmin=200, xmax=1000, label="test accuracy with full feature vectors")
plt.legend()
plt.show()



# In[89]:

fnum_comp[test.index(max(test))]


# In[111]:

from sklearn.svm import SVC
from sklearn.utils import resample
no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
monitor_df = data_df[data_df.severity=='C']
modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
modify_df = resample(modify_df, replace=True, n_samples=2065)
df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
labels_df = df_upsampled[["severity"]]
labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
# TO CHANGE
svd = TruncatedSVD(n_components=250, n_iter=25)
trunc_all_values = svd.fit_transform(upsampled_values)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
accuracy = np.zeros((len(C_range), len(gamma_range)))


# In[112]:

for i in range(len(C_range)):
    c = C_range[i]
    for j in range(len(gamma_range)):
        gamma = gamma_range[j]
        clf = SVC(C=c, gamma=gamma, decision_function_shape='ovo', kernel='rbf')
        data_train, data_test, labels_train, labels_test = train_test_split(trunc_all_values, np.ravel(labels_df.values), test_size=0.2)
        clf.fit(data_train, labels_train)
        preds = clf.predict(data_test)
        test_acc = accuracy_score(labels_test, preds)
        accuracy[i][j] = test_acc
    print("Finished: "+ str(i*len(gamma_range)))
    


# In[113]:

plt.figure(figsize=(8, 6))
plt.imshow(accuracy, interpolation='nearest', cmap=plt.cm.RdPu)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()


# In[117]:

np.ravel(accuracy)[88]


# In[124]:

train = []
test = []
num_trees = []
for i in range(5,30,5):
    randomForest(i)


# In[127]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
def randomForest(num):
    no_action_df = data_df.loc[data_df['severity'].isin(['A','B'])]
    no_action_df = resample(no_action_df, replace=True, n_samples=2065) 
    monitor_df = data_df[data_df.severity=='C']
    modify_df  = data_df.loc[data_df['severity'].isin(['D','X'])]
    modify_df = resample(modify_df, replace=True, n_samples=2065)
    df_upsampled = pd.concat([no_action_df, monitor_df, modify_df])
    upsampled_values = df_upsampled.drop(["severity", "cid_cid"], axis=1).values
    labels_df = df_upsampled[["severity"]]
    labels_df.loc[(labels_df['severity'] == "A") | (labels_df['severity'] == "B"), 'severity'] = "1"
    labels_df.loc[(labels_df['severity'] == "D") | (labels_df['severity'] == "X"), 'severity'] = "3"
    labels_df.loc[(labels_df['severity'] == "C"), 'severity'] = "2"
    # TO CHANGE
    svd = TruncatedSVD(n_components=700, n_iter=25)
    clf = RandomForestClassifier(n_estimators=num)
    trunc_all_values = svd.fit_transform(upsampled_values)
    data_train, data_test, labels_train, labels_test = train_test_split(trunc_all_values, np.ravel(labels_df.values), test_size=0.2)
    clf.fit(data_train, labels_train)
    preds = clf.predict(data_test)
    test_acc = accuracy_score(labels_test, preds)
    train_preds = clf.predict(data_train)
    train_acc = accuracy_score(labels_train, train_preds)
    train.append(train_acc)
    test.append(test_acc)
    print(test_acc)
    print(train_acc)
    print(classification_report(labels_test, preds))
    num_trees.append(num)


# In[125]:

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(range(5,30,5), train, label="train accuracy", color="purple")
plt.plot(range(5,30,5), test, label="test accuracy", color="cyan")
plt.title("Random Forest Hyper-parameter tuning")
plt.ylabel("Validation Accuracy")
plt.xlabel("Number of decision trees")
plt.legend()
plt.show()


# In[128]:

randomForest(20)


# In[ ]:



