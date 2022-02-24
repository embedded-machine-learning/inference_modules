import pytest
import sys
sys.path.append("./")
from pathlib import Path
import os

import hw_modules.rpi4 as rpi4

print(rpi4.__dict__)

__author__ = "Matthias Wess, Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="mobilenetv2-7-sim0.tflite"):
    test_net = os.path.join('tests', 'networks', network)
    rpi4.inference.optimize_network(model_path=test_net, network = "tmp_net", input_shape = [1, 224, 224, 3],
                                    input_node = "data", save_folder = "./tmp")
    assert True

def test_run_network(network="mobilenetv2-7-sim0.tflite"):
    test_net = os.path.join('tests', 'networks', network)
    # test TFLite Interpreter
    rpi4.inference.run_network(tflite_path = test_net, save_dir = "./tmp", niter = 10, print_bool = True, sleep_time=0,
                use_tflite=True, use_pyarmnn=False)
    # test PyArmnn
    rpi4.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                               use_tflite=False, use_pyarmnn=True)
    assert True

def test_read_rpi4_report(network=None):
    # rpi4.parser.read_report(None)
    assert True

def test_rpi4_r2a(network=None):
    #ncs2.parser.r2a(report_file)
    assert True

def test_all(network="mobilenetv2-7-sim0.tflite"):
    test_net = os.path.join('tests', 'networks', network)

    rpi4.inference.optimize_network(model_path=test_net, network = "tmp_net", input_shape = [1, 224, 224, 3],
                                    input_node = "data", save_folder = "./tmp")

    rpi4.inference.run_network(tflite_path = test_net, save_dir = "./tmp", niter = 10, print_bool = True, sleep_time=0,
                use_tflite=True, use_pyarmnn=False)

    rpi4.inference.run_network(tflite_path=test_net, save_dir="./tmp", niter=10, print_bool=True, sleep_time=0,
                               use_tflite=False, use_pyarmnn=True)

    #test_report = Path('tmp','benchmark_average_counters_report.csv')
    rpi4.parser.r2a(None)
    test_rpi4_r2a(None)

    assert True


def main():

    test_optimize_network()
    test_run_network()
    network = "mobilenetv2-7-sim0"
    test_read_rpi4_report(network=network)
    test_all(network='mobilenetv2-7-sim0.tflite')

if __name__ == '__main__':
    main()
