import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import hw_modules.ncs2_ov2019 as ncs2

print(ncs2.__dict__)

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_read_ncs2_report(network="cf_resnet50"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    ncs2.read_report(report_file)

    assert True

#TODO: test fuse Layers -> test delete Layers

def main():
    network = "cf_inceptionv3_imagenet_299_299_11.4G_avg_cnt_rep"
    test_read_ncs2_report(network=network)

if __name__ == '__main__':
    main()
