import math

import numpy as np

def split_patients(patient_admission: dict, admission_codes: dict, code_map: dict, seed=6669) -> (np.ndarray, np.ndarray, np.ndarray):
    print('splitting train, valid, and test pids')
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        # print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission['admission_id']]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    train_num = 5000
    valid_num = 500
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def code_matrix(pids: np.ndarray,
                  patient_admission: dict,
                  admission_codes_encoded: dict, 
                  admission_items_encoded: dict, 
                  max_admission_num: int,
                  code_num: int,
                  item_num: int) -> tuple[np.ndarray, int]:
    print('building train/valid/test admission_code matrix ...')
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=int)    # x = n*q*s, with q -> each visit, s -> codes in each visits
    labs = np.zeros((n, max_admission_num, item_num), dtype=int)
    lens = []
    for i, pid in enumerate(pids):         ### Each patient
        #print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions):         ### Each admission
            admission_id = admission['admission_id']
            code = admission_codes_encoded[admission_id]
            lab = admission_items_encoded[admission_id]
            lab_index = [z-1 for z in lab]
            x[i][k][:len(code)] = code
            labs[i][k][lab_index] = 1
            if k!= 0: lens.append(k)
    #print('\r\t%d / %d' % (len(pids), len(pids)))
    total_possible_examples = sum([len(patient_admission[p])-1 for p in pids])
    return x, labs, np.array(lens), total_possible_examples

def build_code_xy(codes_matrix: np.ndarray,
                  labs_matrix: np.ndarray,
                  n: int,
                  max_admission_num: int,
                  code_num: int,
                  item_num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print('building train/valid/test feature and labels ...')
    x = np.zeros((n, max_admission_num, code_num), dtype=int)    # x = n*q*s, with q -> each visit, s -> codes in each visits
    y = np.zeros((n, code_num), dtype=int)
    labs = np.zeros((n, item_num), dtype=int)
    k = 0
    for i in range(len(codes_matrix)):
        for j in range(1, len(codes_matrix[i])):
            if np.all(codes_matrix[i][j]==0): break
            x[k][:j] = codes_matrix[i][:j]
            y_code_index = [t-1 for t in codes_matrix[i][j] if t!=0]
            y[k][y_code_index] = 1
            labs[k] = labs_matrix[i][j]
            k+=1
        #print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, labs


def build_single_lab_xy(single_patient_admission: np.ndarray,
                  single_admission_items_encoded: np.ndarray,
                  single_admission_codes_encoded: np.ndarray,
                  patient_admission: np.ndarray,
                  admission_codes_encoded: np.ndarray,
                  admission_items_encoded: np.ndarray,
                  n: int,
                  code_num: int,
                  lab_num: int) -> tuple[np.ndarray, np.ndarray]:
    print('building train/valid/test feature and labels for pre-trained labs...')
    x = np.zeros((n, lab_num), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    i = 0
    for pid, admissions in single_patient_admission.items():
        items = single_admission_items_encoded[admissions[0]['admission_id']]
        codes = single_admission_codes_encoded[admissions[0]['admission_id']]
        item_index = [m-1 for m in items]
        code_index = [j-1 for j in codes]
        x[i][item_index] = 1
        y[i][code_index] = 1
        i+=1
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            admission_id = admission['admission_id']
            items = admission_items_encoded[admission_id]
            codes = admission_codes_encoded[admission_id]
            item_index = [m-1 for m in items]
            code_index = [j-1 for j in codes]
            x[i][item_index] = 1
            y[i][code_index] = 1
            i+=1
    #print(i)
    return x, y


def build_heart_failure_y(hf_prefix: str, codes_y: np.ndarray, code_map: dict) -> np.ndarray:
    print('building train/valid/test heart failure labels ...')
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map), ), dtype=int)
    hfs[hf_list - 1] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
