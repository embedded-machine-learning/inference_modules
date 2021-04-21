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

def test_optimize_network(network=""):
    test_net = Path('tests','networks','annette_bench1.pb')
    ncs2.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")

    assert True

def test_run_network(network=""):
    test_net = Path('tests','tmp','annette_bench1.xml')
    ncs2.run_network(test_net)

    assert True

def test_read_ncs2_report(network="cf_inceptionv3_imagenet_299_299_11.4G_avg_cnt_rep"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    ncs2.read_report(report_file)

    assert True

def test_all(network="annette_bench.pb"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    test_net = Path('tests','networks','annette_bench1.pb')
    ncs2.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
    test_net = Path('tests','tmp','annette_bench1.xml')
    ncs2.run_network(test_net)
    test_report = Path('tmp','benchmark_average_counters_report.csv')
    ncs2.read_report(test_report)

    assert True


def main():

    test_optimize_network()
    test_run_network()
    network = "cf_inceptionv3_imagenet_299_299_11.4G_avg_cnt_rep"
    test_read_ncs2_report(network=network)
    test_all(network='annette_bench.pb')

if __name__ == '__main__':
    main()
