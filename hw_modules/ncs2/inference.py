# this script takes a neural network in the intermediate representation .pb file
# converts it to a Movidius NCS2 conform Openvino Intermediate Representation format with .xml and .bin files
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
from powerutils import measurement

try:
    import board
    import digitalio
    print("USB to GPIO adapter pip packages installed")
except:
    print("Pip packages for USB to GPIO adapter not found!")
    print("Without these, the power measurements will not work as intended (only noise will be measured).")


__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def optimize_network(model_path="./models/model.pb", source_fw = "tf", network = "tmp_net", input_shape = [1, 224, 224, 3],
                     input_node = "data", save_folder = "./tmp"):
    mo_file = os.path.join("/", "opt", "intel", "openvino_2021", "deployment_tools", "model_optimizer", "mo.py")

    # check if necessary files exists
    if not os.path.isfile(mo_file):
        logging.info("Model Optimizer not found at:", mo_file)
        return False

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    logging.info("\n**********MOVIDIUS FP16 CONVERSION**********")
    xml_path = ""
    model_name = ""

    if source_fw == "tf":
        # Tensorflow conversion
        # input_shape for tensorflow : batch, width, height, channels
        if input_shape:
            shape = "--input_shape [" + ",".join([str(input_shape[0]), str(input_shape[1]), str(input_shape[2]), str(input_shape[3])]) + "]"
        else:
            shape = ""

        c_conv = ("python3 " + str(mo_file) +
        " --input_model " + str(model_path) +
        " --output_dir " + str(save_folder) +
        " --data_type FP16 " + shape)
        xml_path = os.path.join(save_folder, str(model_path).split(".pb")[0].split("/")[-1]+".xml")
        logging.debug(xml_path)
    elif source_fw == "onnx":
        logging.info("\n**********ONNX CONVERSION**********")

        c_conv = ("python3 " + str(mo_file) +
        " --input_model " + str(model_path) +
        " --output_dir " + str(save_folder) +
        " --data_type FP16")
        xml_path = os.path.join(save_folder, str(model_path).split(".onnx")[0].split("/")[-1] + ".xml")
        logging.debug(xml_path)
    elif source_fw in ["cf", "dk"]:
        # Caffe or Darknet conversion
        logging.info("\n**********CAFFE AND DARKNET CONVERSION NOT SUPPORTED YET**********")
    else:
        logging.info("Model conversion for non standard frameworks is not supported yet!")

    if os.system(c_conv):
        logging.info("\nAn error has occured during conversion!\n")

    logging.info("Openvino Intermediate Representation model generated at: {}".format(xml_path))
    logging.info("\n**********OPENVINO INTERMEDIATE REPRESENTATION MODEL GENERATED AT {}**********".format(xml_path))
    return xml_path


def run_network_new(xml_path = "./tmp/model.xml", report_dir = "./tmp", device = "MYRIAD", niter = 10, print_bool = False, sleep_time=0):
    # initialize power measurement
    pm = measurement.power_measurement(sampling_rate=500000, data_dir=report_dir, max_duration=60, port=1) # port 1 for NCS2
    model_name_kwargs = {"model_name": xml_path.split(".xml")[0].split("/")[-1]}
    ie = IECore()

    led = digitalio.DigitalInOut(board.C0)
    led.direction = digitalio.Direction.OUTPUT
    led.value = True

    statistics = StatisticsReport(StatisticsReport.Config("average_counters", report_dir))

    bin_path = str(xml_path).split(".xml")[0] + ".bin"
    ie_network = ie.read_network(str(xml_path), bin_path)

    input_blob = next(iter(ie_network.input_info))
    out_blob = next(iter(ie_network.outputs))
    n, c, h, w = ie_network.input_info[input_blob].input_data.shape

    exe_network = ie.load_network(str(xml_path), device, config={}, num_requests=1)

    # create report directory if it doesn't exist yet
    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)
    start_time = datetime.utcnow()

    infer_request = exe_network.requests[0]
    # warming up - out of scope
    infer_request.infer()

    # start inference over niter
    infer_requests = exe_network.requests
    times = []

    if sleep_time > 10:
        print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(sleep_time))
        return
    elif sleep_time < 0:
        print("Invalid sleep time {0:.2f}s".format(sleep_time))
        return
    net_input = np.random.randint(0, 255, size=[n, c, h, w])
    pm.start_gather(model_name_kwargs)  # start power measurement
    try:
        for iteration in range(niter): # iterate over inferences
            led.value = False
            infer_requests[0].infer(inputs={input_blob: net_input})
            led.value = True
            if print_bool:
                print("iteration {} took {:.3f} ms".format(iteration, infer_requests[0].latency))
            times.append(infer_requests[0].latency)
            sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")

    # wait the latest inference executions
    status = exe_network.wait()
    if status != StatusCode.OK:
        raise Exception(f"Wait for all requests is failed with status code {status}!")

    sleep(0.02)
    pm.end_gather(True) # end powermeasurement
    print("Power Measurement ended")

    total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
    times.sort()
    if print_bool:
        print("Execution time median: {:.3f} ms".format(median(times)))

    perf_counts = True

    # save performance counters from inference request into list and dump using statistics from Openvino
    # needs rewriting for asynchronous inference requests
    if perf_counts:
        perfs_count_list = []
        perfs_count_list.append(exe_network.requests[0].get_perf_counts())
        if statistics:
            statistics.dump_performance_counters(perfs_count_list)


def run_network(xml_path = "./tmp/model.xml", report_dir = "./tmp", hardware = "MYRIAD", batch = 1, nireq = 1, niter = 10, api = "async"):

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    bench_app_file = os.path.join("/","opt","intel", "openvino_2021", "deployment_tools", "tools",
                                  "benchmark_tool", "benchmark_app.py")
    if not os.path.isfile(bench_app_file):
        logging.info("benchmark_app not found at:", bench_app_file)
        return False

    logging.info("\n**********OPENVINO STARTING INFERENCE**********")

    c_bench = ("python3 " + bench_app_file +
    " -m "  + str(xml_path) +
    " -d " + hardware +
    " -b " + str(batch) +
    " -api " + api +
    " -nireq " + str(nireq) +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + str(report_dir))

    # start inference
    if os.system(c_bench):
        logging.info("An error has occured during benchmarking!")
        return False

    return str(report_dir)+"benchmark_average_counters_report.csv"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-m", '--model_path', help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml', help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='./tmp', help='folder to save the resulting files', required=False)
    parser.add_argument("-d", '--device', default='CPU',  help='device to run inference on', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', type=float, default=0, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument('--nireq', default=1,  help='number of inference requests, used in async mode', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports', help='Directory to save reports into', required=False)
    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()

    if not args.model_path and not args.xml:
        logging.error("Invalid model path passed.")
        sys.exit("Please either pass a frozen pb or an Openvino Intermediate Representation xml model")

    xml_path = args.xml
    if args.model_path:
        # overwrite xml_path during network optimization
        xml_path = optimize_network(args.model_path, source_fw="tf", network="tmp_net", input_shape=[1,416,416,3],
                                    input_node="annette_bench1", save_folder=args.save_folder)

    if  os.path.isfile(xml_path):
        run_network_new(xml_path=xml_path, report_dir="./tmp", device=args.device, niter=args.niter,
                        print_bool=args.print, sleep_time=args.sleep)

    logging.info("\n**********OPENVINO DONE**********")

if __name__ == "__main__":
    main()
