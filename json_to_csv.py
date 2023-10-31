import os
import json
import numpy as np
import pandas as pd
     

path = "ILDC_expert/annotation/"
output_path = "ILDC_expert/Dataset/" ## Give path where you need the converted json files in csv
     

key_to_label = {"ACCEPTED": "ACCEPTED", 
                "Accepted": "ACCEPTED",
                "REJECTED": "REJECTED", 
                "Rejected": "REJECTED",
                "RANK1": "RANK 1",
                "Rank1": "RANK 1",
                "RANK2": "RANK 2",
                "Rank2": "RANK 2",
                "RANK3": "RANK 3",
                "Rank3": "RANK 3",
                "RANK4": "RANK 4",
                "Rank4": "RANK 4",
                "RANK5": "RANK 5",
                "Rank5": "RANK 5"}
     

if not os.path.exists(output_path):
    os.mkdir(output_path)
     
main_rows = []
for case in os.listdir(path):
    curr_path = os.path.join(path, case)
    if not os.path.exists(os.path.join(output_path, case)):
        os.mkdir(os.path.join(output_path, case))
    rows = []
    seen = set()
    for user in os.listdir(curr_path):
        try:
            with open(os.path.join(curr_path, user),"r") as f:
                data = json.load(f)
                f.close()
            for key in data['_referenced_fss'].keys():
                case_text = data['_referenced_fss'][key]['sofaString']
            decision = ""
            for key in list(data['_views']['_InitialView'].keys()):
                if key in key_to_label.keys():
                    for sent in data['_views']['_InitialView'][key]:
                        curr = {}
                        if(case_text[sent['begin']: sent['end']].lower() == "decision"):
                            decision = key_to_label[key]
                            continue
                        curr["TEXT"] = case_text[sent['begin']: sent['end']]
                        curr["SOURCE"] = case_text.replace("\n", "")
                        if key.lower() == 'rank1' or key.lower == 'rank2':
                            curr["IMPORTANT"] = 1
                        elif key.lower() == 'rank3':
                            curr["IMPORTANT"] = 0
                        else:
                            continue
                        if(curr["TEXT"] in seen): 
                            continue
                        else :
                            seen.add(curr["TEXT"]) 
                            rows.append(curr)
        except:
            continue
    for row in rows:
        curr = row
        curr["DECISION"] = decision
        main_rows.append(curr)
    
df = pd.DataFrame(main_rows)
df.to_csv(os.path.join(os.path.join(output_path), "dataset4.csv"))