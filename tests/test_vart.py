import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os

import hw_modules.vart as vart

print(vart.__dict__)

__author__ = "Marco Wuschnig"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network():
    testdict = {}
    #testdict['place_pb_file'] = "./models/annete/annette_bench1.pb"

    testdict['place_pb_file'] = "/home/intel-nuc/marco/marco/inference-modules/hw_modules/vart/models/tf_resnetv150_imagenet_224_224_6.97G_1.3/float/resnet_v1_50_inference.pb"

    testdict['channels'] = 3
    testdict['width'] = 224
    testdict['height'] = 224
    testdict['modelname'] = 'testx'

    
    #test_net = Path('tests','networks',network)
    #vart.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
    #vart.optimize_network(**kwargs)
    vart.optimize_network(**testdict)

    assert True

def test_run_network():

    testdict = {}
    #testdict['place_pb_file'] = "./models/annete/annette_bench1.pb"

    testdict['place_pb_file'] = "./vart/models/tf_resnetv150_imagenet_224_224_6.97G_1.3/float/resnet_v1_50_inference.pb"

    testdict['channels'] = 3
    testdict['width'] = 224
    testdict['height'] = 224
    testdict['modelname'] = 'testx'



    #test_net = Path('tests','tmp',network)
    vart.run_network(**testdict)

    assert True

#def test_read_vart_report(network="benchmark_average_counters_report"):
#    report_file = Path('tests','data','vart',network+'.csv')
#    vart.read_report(report_file)

#    assert True

#def test_vart_r2a(network="benchmark_average_counters_report"):
#    report_file = Path('tests','data','vart',network+'.csv')
#    vart.r2a(report_file)

#    assert True

#def test_all(network="annette_bench.pb"):
#    test_net = Path('tests','networks','annette_bench1.pb')
#    run_args = vart.optimize_network(test_net, source_fw = "tf", network = "tmp_net", image = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
#    report_file = vart.run_network(run_args)
#    vart.r2a(report_file)

#    assert True


def main():



    test_optimize_network()
    test_run_network()


if __name__ == '__main__':
    main()
