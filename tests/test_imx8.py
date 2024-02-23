import pytest
import sys
sys.path.append("./")
from pathlib import Path
import os

import annette.hw_modules.hw_modules.imx8 as imx8

print(imx8.__dict__)

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network=None):
    test_net = network
    imx8.inference.optimize_network(model_path=test_net, network = "tmp_net", input_shape = [1, 224, 224, 3],
                                    input_node = "data", save_folder = "./tmp")
    assert True

def test_run_network(network=None):
    test_net = network
    # test TFLite Interpreter
    imx8.inference.run_network(tflite_path = test_net, save_dir = "./tmp", niter = 10, print_bool = True, sleep_time=0,
                use_tflite=True, use_pyarmnn=False)
    # test PyArmnn
    imx8.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                               use_tflite=False, use_pyarmnn=True)
    assert True

def test_read_imx8_report(network=None):
    # imx8.parser.read_report(None)
    assert True

def test_imx8_r2a(network=None):
    #ncs2.parser.r2a(report_file)
    assert True

def test_all(network=None):
    test_net = network

    imx8.inference.optimize_network(model_path=test_net, network = "tmp_net", input_shape = [1, 224, 224, 3],
                                    input_node = "data", save_folder = "./tmp")

    imx8.inference.run_network(tflite_path = test_net, save_dir = "./tmp", niter = 10, print_bool = True, sleep_time=0,
                use_tflite=True, use_pyarmnn=False)

    imx8.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                               use_tflite=False, use_pyarmnn=True)

    #test_report = Path('tmp','benchmark_average_counters_report.csv')
    imx8.parser.r2a(None)
    test_imx8_r2a(None)

    assert True


def main():
    network = "mobilenetv2-7-sim0"
    test_all(network=network)

if __name__ == '__main__':
    main()
