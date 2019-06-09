
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import json
from os import listdir
from os.path import isfile, join


# In[2]:

with open('cid_to_name.json', 'r') as f:
    cid_to_name = json.load(f)
name_to_cid = {}
for cid, name in cid_to_name.items():
    name_to_cid[name.lower().strip()] = cid
name_to_cid["fluvoxamine"] = name_to_cid["(z) fluvoxamine"]
name_to_cid["cyclosporin"] = name_to_cid["cyclosporin a"]
name_to_cid["fluorouracil"] = name_to_cid["5-fluorouracil"]
name_to_cid["norgestrel"] = name_to_cid["ld norgestrel"]
name_to_cid["methyldopa"] = name_to_cid["dl-methyldopa"]
name_to_cid["methamphetamine"] = name_to_cid["dl-methamphetamine"]
name_to_cid["mitomycin"] = name_to_cid["mitomycin c"]
name_to_cid["cyclosporine"] = name_to_cid["cyclosporin a"]
name_to_cid["vinorelbine"] = name_to_cid["vinorelbine base"]
name_to_cid["fludarabine"] = name_to_cid["fludarabine base"]
name_to_cid["naloxone"] = name_to_cid["naloxone(-)"]
name_to_cid["nasalide®"] = name_to_cid["nasalide"]
name_to_cid["stavudine"] = name_to_cid["stavudine (d4t)"]
name_to_cid["loperamide"] = name_to_cid["loperamide cation"]
name_to_cid["gadodiamide"] = name_to_cid["gadodiamide hydrate"]
name_to_cid["d3 vitamin"] = name_to_cid["1,25-dihydroxyvitamin d3"]
name_to_cid["fluzone"] = name_to_cid["fluconazole"]
name_to_cid["clindamycine"] = name_to_cid["clindamycin"]
name_to_cid["timolol"] = name_to_cid["racemic-timolol"]
name_to_cid["rotigotine"] = name_to_cid["ent-rotigotine"]
name_to_cid["entacapone"] = name_to_cid["cis-entacapone"]


# In[5]:

cid_cid_to_rating = {}
path = "lexi-data2/"
all_lexi_data_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
for file in all_lexi_data_files:
    parse_file(file)


# In[3]:

def parse_file(filename):
    soup = BeautifulSoup(open(filename), "html.parser")
    interaction_list = soup.find_all(class_="intItem")
    for interaction in interaction_list:
        info = interaction.find_all("span")
        if len(info) < 3:
            continue
        rating = info[0].get_text()
        # only get what is before parenthese - remove whitespace
        drug1 = info[1].get_text().split("(")[0].strip().lower()
        drug2 = info[2].get_text().split("(")[0].strip().lower()
        cid1 = get_cid(drug1)
        cid2 = get_cid(drug2)
        if cid1==None or cid2==None:
            print(drug1)
            print(drug2)
            continue
        if int(cid1) > int(cid2):
            # cid1 is always lower
            copy = cid1
            cid1 = cid2
            cid2 = copy
        cid_cid = cid1 + "/"+ cid2
        cid_cid_to_rating[cid_cid] = rating


# In[4]:

def get_cid(drug_name):
    if drug_name in name_to_cid:
        return name_to_cid[drug_name]
    elif drug_name.split()[0] in name_to_cid:
        return name_to_cid[drug_name.split()[0]]
    elif drug_name.replace("cis-", "") in name_to_cid:
        return name_to_cid[drug_name.replace("cis-")]
    else:
        # try just first two
        options = drug_name.split()
        if len(options) > 1:
            name = options[0]+" "+ options[1]
            name = name.strip()
            if name in name_to_cid:
                return name_to_cid[name]
    return None


# In[9]:

import json
json_data = json.dumps(cid_cid_to_rating)
f = open("cid_cid_to_rating2.json","w")
f.write(json_data)
f.close()


# In[7]:

with open('cid_cid_to_sideffects.json', 'r') as f:
    cid_cid_to_sideffects = json.load(f)


# In[48]:

cid_cid_to_sideffects.keys()


# In[8]:

# from the second batch
rating_cids = set(cid_cid_to_rating.keys())
sideeffects_cid = set(cid_cid_to_sideffects.keys())
len(rating_cids.intersection(sideeffects_cid))


# In[12]:

import json
with open('cid_cid_to_rating.json', 'r') as f:
    cid_cid_to_rating_orig = json.load(f)
with open('cid_cid_to_rating2.json', 'r') as f:
    cid_cid_to_rating_new = json.load(f)
cid_cid_to_rating_new.update(cid_cid_to_rating_orig)


# In[14]:

rating_cids = set(cid_cid_to_rating_new.keys())
sideeffects_cid = set(cid_cid_to_sideffects.keys())
len(rating_cids.intersection(sideeffects_cid))


# In[15]:

import json
json_data = json.dumps(cid_cid_to_rating_new)
f = open("cid_cid_to_rating_combined.json","w")
f.write(json_data)
f.close()


# In[7]:

len([rating for cid, rating in cid_cid_to_rating.items() if rating == 'B'])

