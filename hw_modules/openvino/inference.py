# this script takes a neural network in the intermediate representation .pb file
# converts it to a  openvino conform Openvino Intermediate Representation format with .xml and .bin files
# runs inference on the generated model

# Example: python3 inference.py --model_path ../../tests/networks/annette_bench1.pb --save_folder ./tmp/ov_reports/ --device CPU

import logging
import os, sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore, get_version, StatusCode
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, averageCntReport, detailedCntReport
from time import sleep, time
from datetime import datetime
from statistics import median
from pathlib import Path
from yaml.loader import SafeLoader
import yaml

try:
    import board
    import digitalio
    print("USB to GPIO adapter pip packages installed")
except:
    print("Pip packages for USB to GPIO adapter not found!")
    print("Without these, the power measurements will not work as intended (only noise will be measured).")


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


class openvinoClass:
    def __init__(self, config_file="../openvino.yaml"):
        self.config_file = config_file
        f = open(os.path.abspath(config_file), "r")
        data = yaml.load(f, Loader=SafeLoader)
        f.close()
        hw = 'openvino'
        self.ssh_ip = data[hw]['ssh_ip']
        self.ssh_key = data[hw]['ssh_key']
        self.ssh_user = data[hw]['ssh_user']
        self.port = data[hw]['port']
        self.tflite_model = data[hw]['tflite_model']
        self.model_path = data[hw]['model_path']
        self.save_dir = data[hw]['save_dir']
        self.niter = data[hw]['niter']
        self.threads = data[hw]['threads']
        self.bench_file = data[hw]['bench_file']
        self.ssh = data[hw]['ssh']
        self.print = data[hw]['print']
        self.interpreter = data[hw]['interpreter']
        self.pyarmnn = data[hw]['pyarmnn']
        self.sleep = data[hw]['sleep']

    def optimize_network(self, model_path="./models/model.pb", save_folder = "./tmp", **kwargs):
            return {"model_path": model_path,  "save_folder": save_folder}


    def run_network(self, model_path = "./tmp/model.xml", save_folder = "./tmp", device = "CPU", batch = 1, nireq = 1, niter = 10, api = "sync", sleep_time=0, **kwargs):

        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        bench_app_file = os.path.join("benchmark_app")
        # which bench_app_file
        # if operating system is linux
        if sys.platform == "linux" or sys.platform == "linux2":
            stdout = os.popen("which benchmark_app").read()
        elif sys.platform == "win32":
            stdout = os.popen("where.exe benchmark_app.exe").read()
        else:
            print("OS not supported")
            return False
        # if not empty:
        if stdout != "":
            bench_app_file = stdout.strip() 
            print("Using benchmark_app from: ", bench_app_file)
        else:
            return False

        logging.info("\n**********OPENVINO STARTING INFERENCE**********")

        c_bench = (bench_app_file +
        " -m "  + str(model_path) +
        " -d " + device +
        " -b " + str(batch) +
        " -api " + api +
        " -nireq " + str(nireq) +
        " -niter " + str(niter) +
        " --report_type average_counters" +
        " --report_folder " + str(save_folder))

        print(c_bench)
        # start inference
        if os.system(c_bench):
            logging.info("An error has occured during benchmarking!")
            return False
        print("done")

        return str(save_folder)+"/benchmark_average_counters_report.csv"


def main():
    return True

if __name__ == "__main__":
    main()
