import annette.hw_modules.hw_modules.xavier as xavier
import os
from pathlib import Path
import pytest
import sys
sys.path.append("./")


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_optimize_network(network="/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/src/annette/hw_modules/tests/efficientnet_lite1_int8_2.tflite", config_file='config.yaml'):
    test_net = network
    xavierobj = xavier.inference.xavierClass(config_file=config_file)
    xavierobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    assert True

def test_optimize_and_run_network(network="/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/src/annette/hw_modules/tests/efficientnet_lite1_int8_2.tflite", config_file='config.yaml'):
    test_net = network
    xavierobj = xavier.inference.xavierClass(config_file=config_file)
    input = xavierobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    print(input)
    xavierobj.run_network_ssh(**input)
    assert True

def test_run_network(network="/efficientnet_lite1_int8_2_vela.tflite", config_file='config.yaml'):
    test_net = network
    # test TFLite Interpreter
    xavierobj = xavier.inference.xavierClass(config_file=config_file)
    xavierobj.run_network_ssh(tflite_model=test_net, model_path=test_net, niter=10, print_bool=True, sleep_time=0,
                              use_tflite=True, use_pyarmnn=False)
    assert True


def test_read_xavier_report(network=None):
    # xavier.parser.read_report(None)
    assert True


def test_xavier_r2a(network=None):
    # ncs2.parser.r2a(report_file)
    assert True


def test_all(network="/home/ubuntu/imatvey/inference_modules/tests/networks/mobilenetv2-7-sim0.tflite"):
    test_net = network

    xavier.inference.optimize_network(model_path=test_net, network="tmp_net", input_shape=[1, 224, 224, 3],
                                      input_node="data", save_folder="./tmp")

    xavier.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                                 use_tflite=True, use_pyarmnn=False)

    xavier.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                                 use_tflite=False, use_pyarmnn=True)

    # test_report = Path('tmp','benchmark_average_counters_report.csv')
    xavier.parser.r2a(None)
    test_xavier_r2a(None)

    assert True


def main():

    network = "/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/src/annette/hw_modules/tests/matthias_b.onnx"
    res = test_optimize_and_run_network(network=network)

    #net = "efficientnet_lite1_int8_2_vela.tflite"
    #network = "/home/mwess/tmp_tut_merge/SoC_EML_ANNETTE/src/annette/hw_modules/tmp/efficientnet_lite1_int8_2_vela.tflite"
    #test_run_network(network=network)


if __name__ == '__main__':
    main()
