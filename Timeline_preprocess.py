import _pickle as pickle
import os
import numpy as np
from datetime import datetime

# Construct visitfile, gapfile (days), labelfile
def build_timeline(pids, patient_admission, admission_codes):
    visitfile = []
    gapfile = []
    labelfile = []
    for pid in pids:
        admissions = patient_admission[pid]
        for i, admission in enumerate(admissions):
            visit_codes = []
            gap = []
            last_time = datetime.fromisoformat(admission['admission_time'])
            if i == 0:
                continue
            else:
                for j in range(i):
                    admission_time = datetime.fromisoformat(admissions[j]['admission_time'])
                    time_diff = last_time - admission_time
                    gap.append(time_diff.days)
                    visit_codes.append(admission_codes[admissions[j]['admission_id']])
            labelfile.append(admission_codes[admission['admission_id']])
            visitfile.append(visit_codes)
            gapfile.append(gap)
    return visitfile, gapfile, labelfile

if __name__ == "__main__":
    data_path = os.path.join('data', 'mimic3')
    load_path = os.path.join(data_path, 'standard')
    if not os.path.exists(load_path):
        print('please put the pickle files in `data/mimic3/standard`')
        exit()
    patient_data = pickle.load(open(os.path.join(load_path, 'patient.pkl'), 'rb'))
    patient_admission, admission_codes = patient_data['patient_admission'], patient_data['admission_codes']
    pid_data = pickle.load(open(os.path.join(load_path, 'pids.pkl'), 'rb'))
    train_pids, valid_pids, test_pids = pid_data['train_pids'], pid_data['valid_pids'], pid_data['test_pids']

    train_visitfile, train_gapfile, train_labelfile = build_timeline(train_pids, patient_admission, admission_codes)
    valid_visitfile, valid_gapfile, valid_labelfile = build_timeline(valid_pids, patient_admission, admission_codes)
    test_visitfile, test_gapfile, test_labelfile = build_timeline(test_pids, patient_admission, admission_codes)


    timeline_path = os.path.join(data_path, 'timeline')
    if not os.path.exists(timeline_path):
        os.makedirs(timeline_path)
    print("saving visitfile...")
    pickle.dump(train_visitfile, open(os.path.join(timeline_path, 'train_visitfile.npy'), 'wb'))
    pickle.dump(valid_visitfile, open(os.path.join(timeline_path, 'valid_visitfile.npy'), 'wb'))
    pickle.dump(test_visitfile, open(os.path.join(timeline_path, 'test_visitfile.npy'), 'wb'))
    print("saving labelfile...")
    pickle.dump(train_labelfile, open(os.path.join(timeline_path, 'train_labelfile.npy'), 'wb'))
    pickle.dump(valid_labelfile, open(os.path.join(timeline_path, 'valid_labelfile.npy'), 'wb'))
    pickle.dump(test_labelfile, open(os.path.join(timeline_path, 'test_labelfile.npy'), 'wb'))
    print("saving gapfile...")
    pickle.dump(train_gapfile, open(os.path.join(timeline_path, 'train_gapfile.npy'), 'wb'))
    pickle.dump(valid_gapfile, open(os.path.join(timeline_path, 'valid_gapfile.npy'), 'wb'))
    pickle.dump(test_gapfile, open(os.path.join(timeline_path, 'test_gapfile.npy'), 'wb'))
    

