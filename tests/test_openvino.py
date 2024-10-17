import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os
import multiprocessing

import annette.hw_modules.hw_modules.openvino as openvino

print(openvino.__dict__)

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="annette_bench1.pb", config_file='config.yaml'):
    openvinoobj = openvino.inference.openvinoClass(config_file=config_file)
    test_net = Path('tests','networks',network)
    openvinoobj.optimize_network(test_net, source_fw = "tf", network = "tmp_net", input_shape = [1, 416, 416, 3] , input_node = "data", save_folder = "tests/tmp")

    assert True

def test_run_network(network="annette_bench1.pb", config_file='config.yaml'):
    openvinoobj = openvino.inference.openvinoClass(config_file=config_file)
    test_net = Path('tests','tmp',network)
    test_report = openvinoobj.run_network(model_path=network)
    assert True

def test_read_openvino_report(network="benchmark_average_counters_report"):
    report_file = Path('tests','data','openvino',network+'.csv')
    openvino.parser.read_report(report_file)

    assert True

def test_openvino_r2a(network="benchmark_average_counters_report"):
    report_file = Path('tests','data','openvino',network+'.csv')
    openvino.parser.r2a(report_file)

    assert True

def test_all(network="annette_bench.pb", config_file='config.yaml'):
    openvinoobj = openvino.inference.OpenVinoClass(config_file=config_file)
    network = Path('tests','networks','annette_bench1.pb')
    out = openvinoobj.optimize_network(network, source_fw = "tf", network = "tmp_net", input_node = "data", save_folder = "tmp")
    test_report = openvinoobj.run_network(**out)
    openvino.parser.r2a(test_report)

    assert True

def main():
    network='tests/networks/annette_bench1.pb'
    #test_optimize_network(network=network)
    #test_run_network(network=network)
    #test_read_openvino_report(network=network)
    test_all(network='annette_bench.pb')

if __name__ == '__main__':
    main()
