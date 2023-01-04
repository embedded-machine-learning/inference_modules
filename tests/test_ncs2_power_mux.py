import pytest
import sys
sys.path.append("./")
from pathlib import Path
import logging
import os
import multiprocessing
import hw_modules.ncs2 as ncs2
from powerutils import measurement


def test_run_network(network="annette_bench1.xml"):

    test_net = "/home/mivanov/projects/models/yolo/yolov5s_simpl.xml"
    #test_net = "/home/mivanov/projects/inference_modules/tests/ov_networks/annette_bench1.xml"
    kw = {"xml_path": test_net, "report_dir":"./tmp", "device":"MYRIAD"}

    ncs2.inference.run_network_new(**kw)

    assert True


if __name__ == '__main__':
    test_run_network()
