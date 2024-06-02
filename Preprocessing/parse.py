import os
from datetime import datetime

import pandas as pd
import numpy as np


### { subject_ID: [{}, {}, ...] } for patients who have multiple visits
def parse_admission(path) -> dict:
    print('parsing ADMISSIONS.csv ...')
    admission_path = os.path.join(path, 'ADMISSIONS.csv.gz')
    admissions = pd.read_csv(
        admission_path,
        compression = 'gzip', 
        usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'],
        converters={ 'SUBJECT_ID': int, 'HADM_ID': int, 'ADMITTIME': str }
    )
    ### { pid: [ {}, {}, ... ] }
    all_patients = dict()
    for i, row in admissions.iterrows():
        """
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
        """
        pid = row['SUBJECT_ID']
        admission_id = row['HADM_ID']
        #admission_time = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        admission_time = row['ADMITTIME']
        if pid not in all_patients:
            all_patients[pid] = []
        admission = all_patients[pid]
        admission.append({
            'admission_id': admission_id,
            'admission_time': admission_time
        })
    print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

    patient_admission = dict()
    single_patient_admission = dict()
    for pid, admissions in all_patients.items():
        ### Only collect patients with multiple admissions (visits)
        if len(admissions) > 1:
            patient_admission[pid] = sorted(admissions, key=lambda admission: admission['admission_time'])
        else:
            single_patient_admission[pid] = sorted(admissions, key=lambda admission: admission['admission_time'])

    return single_patient_admission, patient_admission


### {admission_id: [ICD9 codes] } for only the corresponding patients who have multiple visits
def parse_diagnoses(path, single_patient_admission: dict, patient_admission: dict) -> dict:
    print('parsing DIAGNOSES_ICD.csv ...')
    diagnoses_path = os.path.join(path, 'DIAGNOSES_ICD.csv.gz')
    diagnoses = pd.read_csv(
        diagnoses_path,
        compression = 'gzip', 
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
        converters={ 'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': str }
    )

    def to_standard_icd9(code: str):
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code

    admission_codes = dict()
    single_admission_codes = dict()
    for i, row in diagnoses.iterrows():
        """
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(diagnoses)), end='')
        """
        pid = row['SUBJECT_ID']
        ### Only interested in patients who have multiple visits
        if pid in patient_admission:
            admission_id = row['HADM_ID']
            code = row['ICD9_CODE']
            if code == '':      ### No diagnosis code for this row
                continue
            code = to_standard_icd9(code)
            if admission_id not in admission_codes:
                codes = []
                admission_codes[admission_id] = codes
            else:
                codes = admission_codes[admission_id]
            if code not in codes: 
                codes.append(code)
            # else:
            #     print(pid, code)
        else:
            if pid in single_patient_admission:
                admission_id = row['HADM_ID']
                code = row['ICD9_CODE']
                if code == '':      ### No diagnosis code for this row
                    continue
                code = to_standard_icd9(code)
                if admission_id not in single_admission_codes:
                    codes = []
                    single_admission_codes[admission_id] = codes
                else:
                    codes = single_admission_codes[admission_id]
                if code not in codes: 
                    codes.append(code)
                # else:
                #     print(pid, code)
            else:
                print(f"Not find pid: {pid}")
    print('\r\t%d in %d rows' % (len(diagnoses), len(diagnoses)))

    return single_admission_codes, admission_codes



def parse_lab(single_patient_admission: dict, patient_admission: dict) -> dict:
    print('parsing LABEVENTS.csv for pre-trained single visitors...')
    path = os.path.dirname(__file__)
    lab_path = os.path.join(path, os.pardir, 'data/mimic3/raw/LABEVENTS.csv.gz')
    labs = pd.read_csv(
        lab_path,
        compression = 'gzip', 
        usecols=['SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'FLAG'],
        converters={ 'SUBJECT_ID': int, 'ITEMID': int, 'CHARTTIME': str, 'FLAG': str})
    single_labs = labs[labs['SUBJECT_ID'].isin(single_patient_admission)]
    multi_labs = labs[labs['SUBJECT_ID'].isin(patient_admission)]
    single_labs = single_labs.to_numpy()
    
    ### {admission: {itme1: [], item2:[]...}, pid2: {...} }
    print("Preparing labs for single visit patients...")
    single_admission_items = dict()
    for i, row in enumerate(single_labs):
        #if i % 10000 == 0:
        #    print('\r\t%d in %d rows' % (i + 1, len(labs)), end='')
        #time = datetime.fromisoformat(row[2])
        time = row[2]
        pid = row[0]
        admission_id = single_patient_admission[pid][0]['admission_id']
        if time >= single_patient_admission[pid][0]['admission_time']: continue
        else:
            itemid = row[1]
            flag = row[3]
            if admission_id not in single_admission_items:
               single_admission_items[admission_id] = dict()
            if itemid in single_admission_items[admission_id]:
                if time > single_admission_items[admission_id][itemid][0]:
                    single_admission_items[admission_id][itemid] = [time, flag]
            else:
                single_admission_items[admission_id][itemid] = [time, flag]
                
    print("Preparing labs for multi visit patients...")           
    ### {admission: [[itme1, time1], [item2, time2]...], admission2: [[]...[]] }
    admission_items = dict()
    for i, (pid, admissions) in enumerate(patient_admission.items()):
        if i % 500 == 0:
            print('\r\t%d in %d patients' % (i + 1, len(patient_admission)))
        target = multi_labs[multi_labs['SUBJECT_ID']==pid]
        ### if labs for this patient at all
        if target.empty: continue
        target = target.sort_values('CHARTTIME', ascending=False)
        for admission in admissions:
            time = admission['admission_time']
            result = target[target['CHARTTIME'] < time]
            ### if no previous labs for this admission
            if result.empty: continue
            result = result.drop_duplicates(['ITEMID'])
            admission_items[admission['admission_id']] = result[['ITEMID', 'FLAG']].to_numpy()
    #multi_labs = multi_labs.to_numpy()
    return single_admission_items, admission_items



def calibrate_patient_by_admission(patient_admission: dict, admission_codes: dict):
    print('calibrating patients by admission ...')
    del_pids = []
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            if admission['admission_id'] not in admission_codes:
                del_pids.append(pid)
                break
    for pid in del_pids:
        admissions = patient_admission[pid]
        for admission in admissions:
            if admission['admission_id'] in admission_codes:
                del admission_codes[admission['admission_id']]
            else:
                print('\tpatient %d have an admission %d without diagnosis' % (pid, admission['admission_id']))
        del patient_admission[pid]
        

        
def calibrate_patient_by_lab(patient_admission: dict, admission_codes: dict, admission_items: dict):
    print('calibrating patients by labs ...')
    del_pids = []
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            if admission['admission_id'] not in admission_items:
                del_pids.append(pid)
                break
    for pid in del_pids:
        admissions = patient_admission[pid]
        for admission in admissions:
            del admission_codes[admission['admission_id']]
            if admission['admission_id'] in admission_items:
                del admission_items[admission['admission_id']]
            #else:
                #print('\tpatient %d have an admission %d without any previous lab' % (pid, admission['admission_id']))
        del patient_admission[pid]

