### For multiple exmaples per patient
import os
import _pickle as pickle

import numpy as np
from sklearn.utils import shuffle

from Preprocessing.parse import parse_admission, parse_diagnoses, parse_lab
from Preprocessing.parse import calibrate_patient_by_admission, calibrate_patient_by_lab
from Preprocessing.encoding import encode_code, encode_lab, adjust_single_visitor_without_codes
from Preprocessing.build_dataset import split_patients, code_matrix
from Preprocessing.build_dataset import build_code_xy, build_single_lab_xy
from Preprocessing.auxiliary import generate_code_levels, generate_patient_code_adjacent, generate_code_code_adjacent, co_occur


if __name__ == '__main__':
    seed = 6669
    data_path = 'data'
    raw_path = os.path.join(data_path, 'mimic3', 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the files in `data/mimic3/raw`')
        exit()
    single_patient_admission, patient_admission = parse_admission(raw_path)
    single_admission_codes, admission_codes = parse_diagnoses(raw_path, single_patient_admission, patient_admission)
    calibrate_patient_by_admission(patient_admission, admission_codes)
    calibrate_patient_by_admission(single_patient_admission, single_admission_codes)
    
    single_admission_items, admission_items = parse_lab(single_patient_admission, patient_admission)
    calibrate_patient_by_lab(single_patient_admission, single_admission_codes, single_admission_items)
    
    print('There are %d valid patients with multiple admissions' % len(patient_admission))       ### 7493
    
    max_admission_num = 0           ### 42
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
    max_code_num_in_a_visit = 0
    for admission_id, codes in admission_codes.items():
        if len(codes) > max_code_num_in_a_visit:
            max_code_num_in_a_visit = len(codes)
    print(f"Maximum number of admissions among multi-visit patients: {max_admission_num}")
    print(f"Maximum number of diagnosis codes for an admission among multi-visit patients: {max_code_num_in_a_visit}")


    ### Encoding
    single_admission_codes_encoded, admission_codes_encoded, code_map = encode_code(single_admission_codes, admission_codes)
    adjust_single_visitor_without_codes(single_patient_admission, single_admission_codes, single_admission_codes_encoded, single_admission_items)  ### 26092 -> 26085
    print('There are %d valid patients with single admission' % len(single_patient_admission))   ### 26085

    single_admission_items_encoded, admission_items_encoded, item_map = encode_lab(single_admission_items, admission_items)

    # Admission with no previous lab records are treated the same as no abnormal records before
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            if admission['admission_id'] not in admission_items:
                admission_items_encoded[admission['admission_id']] = []

    code_num, item_num = len(code_map), len(item_map)     ### 4880, 697
    print(code_num, item_num)

    ### Split patients into training, valid, test
    train_pids, valid_pids, test_pids = split_patients(
        patient_admission,
        admission_codes, code_map)

    
    
    ### Build dataset for diagnosis code
    train_matrix, train_lab_matrix, train_visit_lens, n_train = code_matrix(train_pids, patient_admission, admission_codes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, item_num)
    valid_matrix, valid_lab_matrix, valid_visit_lens, n_valid = code_matrix(valid_pids, patient_admission, admission_codes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, item_num)
    test_matrix, test_lab_matrix, test_visit_lens, n_test = code_matrix(test_pids, patient_admission, admission_codes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, item_num)
    print(n_train, n_valid, n_test)        #8540, 779, 3082
    
    
    train_codes_x, train_codes_y, train_lab_x = build_code_xy(train_matrix, train_lab_matrix, 
                                                                            n_train, max_admission_num, code_num, item_num)
    valid_codes_x, valid_codes_y, valid_lab_x = build_code_xy(valid_matrix, valid_lab_matrix,
                                                                            n_valid, max_admission_num, code_num, item_num)
    test_codes_x, test_codes_y, test_lab_x = build_code_xy(test_matrix, test_lab_matrix, 
                                                                        n_test, max_admission_num, code_num, item_num)
    
    
    ### Build Dataset for pre-trained Lab -> Diagnosis code
    n_pre_trained_examples = len(single_admission_codes_encoded) + sum([len(admission) for p, admission in patient_admission.items()])
    print(n_pre_trained_examples)       ### 45979
    lab_x, lab_y = build_single_lab_xy(single_patient_admission, single_admission_items_encoded, single_admission_codes_encoded, 
                                       patient_admission, admission_codes_encoded, admission_items_encoded, 
                                       n_pre_trained_examples, code_num, item_num)
    
    lab_x, lab_y = shuffle(lab_x, lab_y, random_state=seed)
    train_single_lab_x, train_single_lab_y = lab_x[:int(n_pre_trained_examples*0.85)], lab_y[:int(n_pre_trained_examples*0.85)]
    valid_single_lab_x, valid_single_lab_y = lab_x[int(n_pre_trained_examples*0.85):-int(n_pre_trained_examples*0.1)], lab_y[int(n_pre_trained_examples*0.85):-int(n_pre_trained_examples*0.1)]
    test_single_lab_x, test_single_lab_y = lab_x[-int(n_pre_trained_examples*0.1):], lab_y[-int(n_pre_trained_examples*0.1):]
    ############
    
    
    print(train_codes_x.shape, train_codes_y.shape, train_lab_x.shape, train_visit_lens.shape)  # (8540, 42, 4880) (8540, 4880) (8540, 697) (8540,)
    print(valid_codes_x.shape, valid_codes_y.shape, valid_lab_x.shape, valid_visit_lens.shape)  # (779, 42, 4880) (779, 4880) (779, 697) (779,)
    print(test_codes_x.shape, test_codes_y.shape, test_lab_x.shape, test_visit_lens.shape)      # (3082, 42, 4880) (3082, 4880) (3082, 697) (3082,)
   

    ### For CGL
    code_levels = generate_code_levels(data_path, code_map)
    patient_code_adj = generate_patient_code_adjacent(code_x=train_codes_x, code_num=code_num)
    code_code_adj_t = generate_code_code_adjacent(code_level_matrix=code_levels, code_num=code_num)
    co_occur_matrix = co_occur(train_pids, patient_admission, admission_codes_encoded, code_num)
    code_code_adj = code_code_adj_t * co_occur_matrix
    


    ### Save the Data
    train_codes_data = (train_codes_x, train_codes_y, train_lab_x, train_visit_lens)
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_lab_x, valid_visit_lens)
    test_codes_data = (test_codes_x, test_codes_y, test_lab_x, test_visit_lens)
    train_labs_data = (train_single_lab_x, train_single_lab_y)
    valid_labs_data = (valid_single_lab_x, valid_single_lab_y)
    test_labs_data = (test_single_lab_x, test_single_lab_y)
    
    
    
    mimic3_path = os.path.join('data', 'mimic3')
    encoded_path = os.path.join(mimic3_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    
    
    print('saving encoded data ...')
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    
    print('saving standard data ...')
    standard_path = os.path.join(mimic3_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_labs_data': train_labs_data,
        'valid_labs_data': valid_labs_data,
        'test_labs_data': test_labs_data
    }, open(os.path.join(standard_path, 'labs_dataset.pkl'), 'wb'))
    pickle.dump({
        'code_levels': code_levels,
        'patient_code_adj': patient_code_adj,
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))
    pickle.dump({
        'patient_admission': patient_admission,
        'admission_codes': admission_codes,
    }, open(os.path.join(standard_path, 'patient.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(standard_path, 'pids.pkl'), 'wb'))
    