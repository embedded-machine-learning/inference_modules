import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os
import multiprocessing

import hw_modules.ncs2 as ncs2

print(ncs2.__dict__)

__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def test_optimize_network(network="annette_bench1.pb"):
    test_net = Path('tests','networks',network)
    ncs2.inference.optimize_network(test_net, source_fw = "tf", network = "tmp_net", input_shape = [1, 416, 416, 3] , input_node = "data", save_folder = "tests/tmp")

    assert True

def test_run_network(network="annette_bench1.xml"):
    test_net = Path('tests','tmp',network)
    #ncs2.inference.run_network_new(test_net, report_dir = "./tests/data/ncs2_ov2019")
    #test_net = "/home/intel-nuc/imatvey/yolov5/models/yolov5s_simpl_2021.4.xml"
    #test_net = "tests/ov_networks/annette_bench1.xml"
    kw = {"xml_path": test_net, "report_dir":"./tmp", "device":"MYRIAD"}
    p = multiprocessing.Process(target=ncs2.inference.run_network_new,kwargs=kw)
    p.start()
    #(xml_path=test_net, report_dir="./tmp", device="MYRIAD", print_bool=True)

    # Wait for 10 seconds or until process finishes
    p.join(3)

    # If thread is still active
    if p.is_alive():
        print("running... let's kill it...")

        # Terminate - may not work if process is stuck for good
        p.terminate()
        # OR Kill - will work for sure, no chance for process to finish nicely however
        #p.kill()

        p.join()

    print("over")

    assert True

def test_read_ncs2_report(network="benchmark_average_counters_report"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    ncs2.parser.read_report(report_file)

    assert True

def test_ncs2_r2a(network="benchmark_average_counters_report"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    ncs2.parser.r2a(report_file)

    assert True

def test_all(network="annette_bench.pb"):
    report_file = Path('tests','data','ncs2_ov2019',network+'.csv')
    test_net = Path('tests','networks','annette_bench1.pb')
    ncs2.inference.optimize_network(test_net, source_fw = "tf", network = "tmp_net", input_shape = [1, 1, 1, 3] , input_node = "data", save_folder = "tests/tmp")
    test_net = Path('tests','tmp','annette_bench1.xml')
    ncs2.inference.run_network(test_net)
    test_report = Path('tmp','benchmark_average_counters_report.csv')
    ncs2.parser.r2a(test_report)

    assert True

def main():

    test_optimize_network()
    test_run_network()
    network = "cf_inceptionv3_imagenet_299_299_11.4G_avg_cnt_rep"
    test_read_ncs2_report(network=network)
    test_all(network='annette_bench.pb')

if __name__ == '__main__':
    main()
