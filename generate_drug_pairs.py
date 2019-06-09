
# coding: utf-8

# In[1]:

import pandas as pd
import urllib.request
import json


# In[3]:

data = pd.read_csv("ChChSe-Decagon_polypharmacy.csv")
cid_cid_groups = data.groupby(["# STITCH 1"])["STITCH 2"].apply(set)


# In[2]:

with open('cid_cidgroupnames.json', 'r') as f:
    cid_cidgroupnames = json.load(f)
with open('cid_to_name.json', 'r') as f:
    cid_to_name = json.load(f)
print(len(cid_cidgroupnames))


# In[11]:

data.head(5)


# In[53]:

# determine groups to enter into lexi-interact to get interactions
index = 0
for cid1, cid2_list in cid_cid_groups.iteritems():
    cid1 = getCompoundNameFromCID(cid1.replace("CID", ""))
    if cid1 in cid_cidgroupnames:
        continue
    if cid1 == None:
        continue
    cid_cidgroupnames[cid1] = []
    for cid2 in cid2_list:
        cid2 = getCompoundNameFromCID(cid2.replace("CID", ""))
        if cid2 == None:
            continue
        cid_cidgroupnames[cid1].append(cid2)
    if index%10 == 0:
        print("Finished: " + str(index))
    index += 1


# In[4]:

import json
json_data = json.dumps(cid_cidgroupnames)
f = open("cid_cidgroupnames.json","w")
f.write(json_data)
f.close()
json_data = json.dumps(cid_to_name)
f = open("cid_to_name.json","w")
f.write(json_data)
f.close()


# In[12]:

# determine side effects of cid pairs
# that will be turned into feature vectors
cid_cid_to_sideffects = {}
side_effect_code_to_name = {}
side_effect_code_to_index = {}
index = 0
for index, row in data.iterrows():
    cid1 = row["# STITCH 1"].replace("CID", "")
    cid2 = row["STITCH 2"].replace("CID", "")
    side_effect_code = row["Polypharmacy Side Effect"]
    side_effect_name = row["Side Effect Name"]
    side_effect_code_to_name[side_effect_code] = side_effect_name
    if side_effect_code not in side_effect_code_to_index:
        side_effect_code_to_index[side_effect_code] = int(index)
        index += 1
    if cid1 == cid2:
        # only care about drug interactions
        continue
    if int(cid1) > int(cid2):
        # cid1 is always lower
        copy = cid1
        cid1 = cid2
        cid2 = copy
    cid_cid = cid1 + "/" +cid2
    if cid_cid not in cid_cid_to_sideffects:
        cid_cid_to_sideffects[cid_cid] = []
    cid_cid_to_sideffects[cid_cid].append(side_effect_code)


# In[13]:

side_effect_code_to_index['C0022660']


# In[29]:

import json
json_data = json.dumps(cid_cid_to_sideffects)
f = open("cid_cid_to_sideffects.json","w")
f.write(json_data)
f.close()
json_data = json.dumps(side_effect_code_to_name)
f = open("side_effect_code_to_name.json","w")
f.write(json_data)
f.close()
json_data = json.dumps(side_effect_code_to_index)
f = open("side_effect_code_to_index","w")
f.write(json_data)
f.close()


# In[47]:

def getCompoundNameFromCID(cid):
    if cid in cid_to_name:
        return cid_to_name[cid]
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"+str(cid)+"/synonyms/json"
    try:
        contents = urllib.request.urlopen(url)
    except urllib.error.HTTPError as err:
        return None
    except urllib.error.URLError as err:
        print(err.reason)
    data = json.load(contents)
    if "InformationList" in data:
        if "Information" in data["InformationList"]:
            if "Synonym" in data["InformationList"]["Information"][0]:
                cid_to_name[cid] = data["InformationList"]["Information"][0]["Synonym"][0]
                return data["InformationList"]["Information"][0]["Synonym"][0]
    cid_to_name[cid] = None
    return None


# In[120]:

drug_names = sorted(cid_cidgroupnames)
print(drug_names[310])
print(cid_cidgroupnames[drug_names[310]])    


# In[22]:

side_effect_code_to_index = {}
index = int(0)
for side_effect_code in side_effect_code_to_name:
    if side_effect_code not in side_effect_code_to_index:
        side_effect_code_to_index[side_effect_code] = int(index)
        index += 1


# In[25]:

import numpy as np
# generate feature vectors
cid_cid_to_feature_vectors = {}
num_side_effects = len(side_effect_code_to_name)
for cid_cid in cid_cid_to_sideffects:
    cid_cid_to_feature_vectors[cid_cid] = list(np.zeros(num_side_effects))
    for side_effect_code in cid_cid_to_sideffects[cid_cid]:
        side_effect_index = side_effect_code_to_index[side_effect_code]
        cid_cid_to_feature_vectors[cid_cid][side_effect_index] += 1


# In[28]:

import json
json_data = json.dumps(cid_cid_to_feature_vectors)
f = open("cid_cid_to_feature_vectors.json","w")
f.write(json_data)
f.close()
json_data = json.dumps(side_effect_code_to_index)
f = open("side_effect_code_to_index.json","w")
f.write(json_data)
f.close()


# In[27]:

len(cid_cid_to_feature_vectors)


# In[ ]:



