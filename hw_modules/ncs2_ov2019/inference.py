# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

import argparse
import os, sys, threading
from os import system
from sys import stdout
from time import sleep, time
import numpy as np
import logging

def optimize_network(pb, source_fw = "tf", network = "tmp_net", image = [1, 224, 224, 3] , input_node = "data", save_folder = "tests/tmp"):
    mo_file = os.path.join("/", "opt", "intel", "openvino",
    "deployment_tools", "model_optimizer", "mo.py")

    pb = str(pb)

    # check if necessary files exists
    if not os.path.isfile(mo_file):
        print("model optimizer not found at:", mo_file)

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    print("\n**********Movidius FP16 conversion**********")
    xml_path = ""
    model_name = ""

    if source_fw == "tf":
        # Tensorflow conversion
        # input_shape for tensorflow : batch, width, height, channels
        shape = "["+str(image[0])+","+str(image[1])+","+str(image[2])+","+str(image[3])+"]"

        c_conv = ("python3 " + mo_file +
        " --input_model " + pb +
        " --output_dir " + save_folder +
        " --data_type FP16 " +
        " --input_shape " + shape)
        xml_path = os.path.join(save_folder, pb.split(".pb")[0].split("/")[-1]+".xml")
        logging.debug(xml_path)
    elif source_fw in ["cf", "dk"]:
        # Caffe or Darknet conversion
        # input_shape : batch, channels, width, height
        input_proto =  pb.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
        shape = "["+str(image[0])+","+str(image[2])+","+str(image[3])+","+str(image[1])+"]"

        if "SPnet" in pb:
            input_node = "demo"
        else:
            input_node = "data"

        c_conv = ("python3 " + mo_file +
        " --input_model " + pb +
        #" --input_proto " + input_proto +
        " --output_dir " + save_folder +
        " --data_type FP16 " +
        " --input_shape " + shape +
        " --input " + input_node) # input node sometimes called demo)
    
    if os.system(c_conv):
        print("\nAn error has occured during conversion!\n")


def run_network(report_dir):

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    bench_app_file = os.path.join("/","opt","intel", "openvino",
    "deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")
    if not os.path.isfile(bench_app_file):
        print("benchmark_app not found at:", bench_app_file)

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
    " -d MYRIAD " +
    " -b 1 " +
    " -api " + api +
    " -nireq " + str(nireq) +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + report_dir)

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    new_avg_bench_path = os.path.join(report_dir, "_".join(("bacr", model_name.split(".pb")[0], str(index_pm), api,
                                                          "n" + str(niter), "ni" + str(nireq) + ".csv")))
    new_stat_rep_path = os.path.join(report_dir, "_".join(("stat_rep", model_name.split(".pb")[0], str(index_pm), api,
                                                           "n" + str(niter), "ni" + str(nireq) + ".csv")))
    # rename the default report file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"), new_avg_bench_path)
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_report.csv"), new_stat_rep_path)

    bench_over = True # this ends the power data gathering
    print("**********REPORTS GATHERED**********")

    return new_avg_bench_path, new_stat_rep_path

def run_bench(daq_device, low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags,
              data, data_dir, data_fname,  power_measurement, index_pm,
              xml = "", pb = "",save_folder = "./tmp", report_dir = "report", niter = 100, api = "sync", proto="", nireq=1):

    global bench_over


    # start measurement in parallel to inference
    #daq_measurement(low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags, data)

    if uldaq_import and power_measurement == "True":
        x = threading.Thread(target=daq_measurement, args=(daq_device, low_channel, high_channel, input_mode,
                        ranges, samples_per_channel,
                        rate, scan_options, flags, data, data_dir, data_fname, index_pm, api, niter, nireq))
        x.start()

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
    " -d MYRIAD " +
    " -b 1 " +
    " -api " + api +
    " -nireq " + str(nireq) +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + report_dir)

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    new_avg_bench_path = os.path.join(report_dir, "_".join(("bacr", model_name.split(".pb")[0], str(index_pm), api,
                                                          "n" + str(niter), "ni" + str(nireq) + ".csv")))
    new_stat_rep_path = os.path.join(report_dir, "_".join(("stat_rep", model_name.split(".pb")[0], str(index_pm), api,
                                                           "n" + str(niter), "ni" + str(nireq) + ".csv")))
    # rename the default report file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"), new_avg_bench_path)
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_report.csv"), new_stat_rep_path)

    bench_over = True # this ends the power data gathering
    print("**********REPORTS GATHERED**********")

    return new_avg_bench_path, new_stat_rep_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-p", '--pb', default='yolov3.pb',
                        help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml', default='yolov3.xml',
                        help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='save',
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-a", '--api', default='sync',
                        help='synchronous or asynchronous mode [sync, async]',
                        required=False)
    parser.add_argument("-n", '--niter', default=10,
                        help='number of iterations', required=False)
    parser.add_argument('--nireq', default=1,
                        help='number of inference requests, useful in async mode', required=False)
    parser.add_argument("-pr", '--proto', default='caffe.proto',
                        help='Prototext for Xilinx Caffe', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports',
                        help='Directory to save reports into', required=False)
    parser.add_argument("-pm", '--power_measurement', default='False',
                        help='parse "True" when conducting power measurements', required=False)
    args = parser.parse_args()

    index_run = 0

    if not args.pb and not args.xml:
        sys.exit("Please pass either a frozen pb or IR xml/bin model")
    
    run_bench(xml=args.xml, pb=args.pb, save_folder=args.save_folder,
              report_dir=args.report_dir, niter=args.niter,
              api=args.api, proto=args.proto, nireq=args.nireq)
