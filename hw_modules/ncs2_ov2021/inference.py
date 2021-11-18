# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

import argparse
import os, sys, threading
from os import system
import numpy as np
import logging

def optimize_network(pb, source_fw = "tf", network = "tmp_net", image = [1, 224, 224, 3] , input_node = "data", save_folder = "./tmp"):
    mo_file = os.path.join("/", "opt", "intel", "openvino_2021",
    "deployment_tools", "model_optimizer", "mo.py")

    pb = str(pb)

    # check if necessary files exists
    if not os.path.isfile(mo_file):
        logging.info("model optimizer not found at:", mo_file)
        assert False

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    logging.info("\n**********Movidius FP16 conversion**********")
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
        logging.info("\nAn error has occured during conversion!\n")
        assert False

    logging.info(xml_path)
    
    return xml_path

def run_network(xml_path = None, report_dir = "./tmp", hardware = "MYRIAD", batch = 1, nireq = 1, niter = 10, api = "async"):

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    bench_app_file = os.path.join("/","opt","intel", "openvino_2021",
    "deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")
    if not os.path.isfile(bench_app_file):
        logging.error("benchmark_app not found at:", bench_app_file)
        assert False

    c_bench = ("python3 " + bench_app_file +
    " --path_to_model " + str(xml_path) +
    " -d " + hardware +
    " -b " + str(batch) +
    " -api " + api +
    " -nireq " + str(nireq) +
    " -niter " + str(niter) +
    " --perf_counts " +
    " --report_type average_counters " +
    " --report_folder " + report_dir)

    # start inference
    logging.info("Executing command: " + c_bench)
    os.system(c_bench)
    return
    if os.system(c_bench):
        logging.info("An error has occured during benchmarking!")
        assert False


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-p", '--pb',
                        help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml',
                        help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='./tmp',
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-a", '--api', default='async',
                        help='synchronous or asynchronous mode [sync, async]',
                        required=False)
    parser.add_argument("-b", '--batch_size', default=1,
                        help='batch size, typically 1', required=False)
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
    xml_path = args.xml

    if not args.pb and not args.xml:
        logging.error("Invalid model path passed.")
        sys.exit("Please either pass a frozen pb or an IR xml/bin model")

    if args.pb:
        xml_path = optimize_network(args.pb, source_fw="tf", network="tmp_net", image=[1, 416, 416, 3],
                                    input_node="inputs",
                                    save_folder=args.save_folder)
        print(xml_path)

    if xml_path:
        run_network(xml_path=xml_path, report_dir="./reps", hardware="MYRIAD", batch=1, nireq=2, niter=10, api=args.api)

    # TODO reattach the converter inference etc


if __name__ == "__main__":
    main()