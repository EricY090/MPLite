import re

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def encode_code(single_admission_codes: dict, admission_codes: dict) -> tuple[dict, dict]:
    print('encoding diagnosis code ...')
    
    ### Map each ICD9 code to a number
    code_map = dict()
    for i, (admission_id, codes) in enumerate(admission_codes.items()):
        for code in codes:
            if code not in code_map:
                code_map[code] = len(code_map) + 1
                
    # for j, (single_admission_id, single_codes) in enumerate(single_admission_codes.items()):
    #     for code in single_codes:
    #         if code not in code_map:
    #             print(code)
    #             code_map[code] = len(code_map) + 1
    

    ### {Admission: [mapped number for ICD9 codes]}
    admission_codes_encoded = {
        admission_id: [code_map[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    single_admission_codes_encoded = {}
    for admission_id, codes in single_admission_codes.items():
        single_admission_codes_encoded[admission_id] = []
        for code in codes:
            if code in code_map:
                single_admission_codes_encoded[admission_id].append(code_map[code])
    
    return single_admission_codes_encoded, admission_codes_encoded, code_map


def adjust_single_visitor_without_codes(single_patient_admission, single_admission_codes, single_admission_codes_encoded, single_admission_items):
    del_pids = []
    for pid, admissions in single_patient_admission.items():
        if len(single_admission_codes_encoded[admissions[0]['admission_id']]) == 0:
            del_pids.append(pid)
    for pid in del_pids:
        print(f"Patient ID: {pid} has an admission with no codes in code map.")
        del single_admission_codes[single_patient_admission[pid][0]['admission_id']]
        del single_admission_items[single_patient_admission[pid][0]['admission_id']]
        del single_admission_codes_encoded[single_patient_admission[pid][0]['admission_id']]
        del single_patient_admission[pid]



def encode_lab(single_admission_items: dict, admission_items: dict) -> tuple[dict, dict]:
    print('encoding lab ...')
    
    ### Map each itemid to a number
    ### {Patient: [mapped number for itemid]}
    item_map = dict()
    single_admission_items_encoded = dict()
    admission_items_encoded = dict()
    for i, (admission, items) in enumerate(single_admission_items.items()):
        if admission not in single_admission_items_encoded:
            single_admission_items_encoded[admission] = []
        else: print("Impossible")
        for itemid, l in items.items():
            if itemid not in item_map:
                item_map[itemid] = len(item_map) + 1
            if l[1] == "abnormal":
                single_admission_items_encoded[admission].append(item_map[itemid])
    for i, (admission, items) in enumerate(admission_items.items()):
        if admission not in admission_items_encoded:
            admission_items_encoded[admission] = []
        else: print("Impossible")
        for item in items:
            if item[0] not in item_map:
                item_map[item[0]] = len(item_map) + 1
            if item[1] == "abnormal":
                admission_items_encoded[admission].append(item_map[item[0]])
    
    return single_admission_items_encoded, admission_items_encoded, item_map
