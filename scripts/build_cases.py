import os
import sys
sys.path.append(os.path.abspath('..'))
import subprocess
from modules import io

CASES_DIR = os.path.join('.','data', 'cases')
RESULTS_DIR = os.path.join('.','results')

configs = [
'i2i_regression_ct.yaml',
'i2i_regression_mr.yaml',
'rf_2_ct.yaml',
'rf_2_mr.yaml'
]

for c in configs:
    c_ = os.path.join('.','config',c)

    c_yml = io.load_yaml(c_)

    test_patterns = c_yml['TEST_PATTERNS']

    for t in test_patterns:
        case_file = "case.{}.yml".format(t)
        case_file = os.path.join(CASES_DIR, case_file)

        case = io.load_yaml(case_file)

        image = case['IMAGE']
        name  = case['NAME']
        paths = case['PATHS']

        for f in [image,paths]:
            if not os.path.exists(f): raise RuntimeError("{} does not exist".format(f))

        print("building {} - {}".format(name, c))

        subprocess.check_call('python build_cv_model.py -i {} -p {} -c {} -o {} -n {}'.format(
            image, paths, c_, RESULTS_DIR, name
        ))
