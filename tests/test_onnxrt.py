import annette.hw_modules.hw_modules.onnxrt as onnxrt
import onnxruntime as rt
import os
from pathlib import Path
import pytest
import sys
sys.path.append("./")


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def test_optimize_network(network=None, config_file='config.yaml'):
    test_net = network
    onnxrtobj = onnxrt.inference.onnxrtClass(config_file=config_file)
    onnxrtobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    assert True

def test_optimize_and_run_network(network=None, config_file='config.yaml'):
    test_net = network
    onnxrtobj = onnxrt.inference.onnxrtClass(config_file=config_file)
    input = onnxrtobj.optimize_network(model_path=test_net, network=network, input_shape=[1, 224, 224, 3],
                               input_node="data", save_folder="./tmp")
    print(input)
    onnxrtobj.run_network_ssh(**input)
    assert True

def test_run_network(network=None, config_file='config.yaml'):
    test_net = network
    # test TFLite Interpreter
    onnxrtobj = onnxrt.inference.onnxrtClass(config_file=config_file)
    onnxrtobj.run_network(tflite_model=test_net, model_path=test_net, niter=10, print_bool=True, sleep_time=0,
                              use_tflite=True, use_pyarmnn=False)
    assert True

def test_onnxrt_r2a(network=None):
    # ncs2.parser.r2a(report_file)
    assert True

def test_all(network=None, config_file='config.yaml'):
    test_net = network
    onnxrtobj = onnxrt.inference.onnxrtClass(config_file=config_file)
    onnxrtobj.optimize_network(model_path=test_net, network="tmp_net", save_folder="./tmp")
    report = onnxrtobj.run_network(model_path=test_net, save_dir="./tmp")
    onnxrt.parser.r2a(report)
    assert True


def main():

    network = "tests/networks/yolov8n-cls.onnx"
    test_all(network=network)

if __name__ == '__main__':
    main()
