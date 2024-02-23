import annette.hw_modules.hw_modules.tf_basic as tf_basic
import os
from pathlib import Path
import pytest
import sys
sys.path.append("./")

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_optimize_network(network="", config_file='config.yaml'):
    test_net = network
    tf_basicobj = tf_basic.inference.tf_basicClass(config_file=config_file)
    tf_basicobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    assert True

def test_optimize_and_run_network(network="", config_file='config.yaml'):
    test_net = network
    tf_basicobj = tf_basic.inference.tf_basicClass(config_file=config_file)
    input = tf_basicobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    print(input)
    tf_basicobj.run_network(**input)
    assert True

def test_run_network(network=None, config_file='config.yaml'):
    test_net = network
    # test TFLite Interpreter
    tf_basicobj = tf_basic.inference.tf_basicClass(config_file=config_file)
    tf_basicobj.run_network(model_path=test_net)
    assert True

def test_tf_basic_r2a(report_file=None):
    tf_basic.parser.r2a(report_file)
    assert True

def test_all(network=None, config_file='config.yaml'):
    test_net = network
    tf_basicobj = tf_basic.inference.tf_basicClass(config_file=config_file)
    tf_basicobj.optimize_network(model_path=test_net, network="tmp_net", input_shape=[1, 224, 224, 3],
                                      input_node="data", save_folder="./tmp")

    test = tf_basicobj.run_network(model_path=test_net, save_folder="./tmp")
    print(test)

    # test_report = Path('tmp','benchmark_average_counters_report.csv')
    tf_basic.parser.r2a(test)
    assert True


def main():

    network = "tests/networks/annette_bench1.pb"
    #test_run_network(network=network)
    test_all(network=network)

if __name__ == '__main__':
    main()
