import holoclean
from detect import NullDetector, ViolationDetector, OutofDomainDetector
from repair.featurize import *

# 0. Init Files.
data = 'testdata/hospital.csv'
clean_data = 'testdata/hospital_clean.csv'
constraints = 'testdata/hospital_constraints.txt'

# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    db_host='localhost',
    db_name='holo',
    db_user='postgres',
    db_pwd='1234567890',
    domain_thresh_1=0,
    domain_thresh_2=0,
    weak_label_thresh=0.99,
    max_domain=10000,
    cor_strength=0.6,
    nb_cor_strength=0.8,
    epochs=10,
    weight_decay=0.01,
    learning_rate=0.001,
    threads=1,
    batch_size=1,
    timeout=3*60000,
    feature_norm=False,
    weight_norm=False,
    optimizer='adam',
    verbose=True,
    bias=False,
    print_fw=True,
    debug_mode=False
).session

# 2. Load training data and denial constraints.
hc.load_data('hospital', data)
# hc.load_dcs(constraints)
# hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), OutofDomainDetector()]
hc.detect_errors(detectors)

# , ViolationDetector()

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [
    InitAttrFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    # ConstraintFeaturizer(),
]
hc.repair_errors(featurizers)

# 5. Evaluate the correctness of the results.
print(hc.evaluate(fpath=clean_data,
            tid_col='tid',
            attr_col='attribute',
            val_col='correct_val'))