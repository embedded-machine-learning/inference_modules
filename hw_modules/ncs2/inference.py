# this script takes a neural network in the intermediate representation .pb file
# converts it to a Movidius NCS2 conform Openvino Intermediate Representation format with .xml and .bin files
# runs inference on the generated model

# Example: python3 inference.py --model_path ../../tests/networks/annette_bench1.pb --save_folder ./tmp/ov_reports/ --device CPU

import logging
import os, sys


__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def optimize_network(model_path="./models/model.pb", source_fw = "tf", network = "tmp_net", input_shape = [1, 224, 224, 3] , input_node = "data", save_folder = "./tmp"):
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
        xml_path = os.path.join(save_folder, model_path.split(".pb")[0].split("/")[-1]+".xml")
        logging.debug(xml_path)
    elif source_fw in ["cf", "dk"]:
        # Caffe or Darknet conversion
        logging.info("\n**********CAFFE AND DARKNET CONVERSION NOT SUPPORTED YET**********")
        """
        # input_shape : batch, channels, width, height
        input_proto =  model_path.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
        shape = "["+str(input_shape[0])+","+str(input_shape[3])+","+str(input_shape[1])+","+str(input_shape[2])+"]"

        if "SPnet" in model_path:
            input_node = "demo"
        else:
            input_node = "data"

        c_conv = ("python3 " + mo_file +
        " --input_model " + model_path +
        #" --input_proto " + input_proto +
        " --output_dir " + save_folder +
        " --data_type FP16 " +
        " --input_shape " + shape +
        " --input " + input_node) # input node sometimes called demo)
        xml_path = os.path.join(save_folder, model_path.split(".caffemodel")[0].split("/")[-1] + ".xml")
        logging.debug(xml_path)"""
    else:
        logging.info("Model conversion for non standard frameworks is not supported yet!")

    if os.system(c_conv):
        logging.info("\nAn error has occured during conversion!\n")
        return False

    logging.info("Openvino Intermediate Representation model generated at:", xml_path)
    logging.info("\n**********OPENVINO INTERMEDIATE REPRESENTATION MODEL GENERATED AT {}**********".format(xml_path))
    return xml_path


def run_network(xml_path = "./tmp/model.xml", report_dir = "./tmp", hardware = "CPU", batch = 1, nireq = 1, niter = 10, api = "async"):

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    bench_app_file = os.path.join("/","opt","intel", "openvino_2021", "deployment_tools", "tools",
                                  "benchmark_tool", "benchmark_app.py")
    if not os.path.isfile(bench_app_file):
        logging.info("benchmark_app not found at:", bench_app_file)
        return False

    logging.info("\n**********OPENVINO STARTING INFERENCE**********")

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
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
    parser.add_argument("-m", '--model_path', default='yolov3.pb',
                        help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml', default='yolov3.xml',
                        help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='./tmp',
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-a", '--api', default='async',
                        help='synchronous or asynchronous mode [sync, async]',
                        required=False)
    parser.add_argument("-d", '--device', default='CPU',
                        help='device to run inference on',
                        required=False)
    parser.add_argument("-b", '--batch_size', default=1,
                        help='batch size', required=False)
    parser.add_argument("-n", '--niter', default=10,
                        help='number of iterations', required=False)
    parser.add_argument('--nireq', default=1,
                        help='number of inference requests, used in async mode', required=False)
    parser.add_argument("-pr", '--proto', default='caffe.proto',
                        help='Prototext for Xilinx Caffe', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports',
                        help='Directory to save reports into', required=False)
    args = parser.parse_args()

    if not args.model_path and not args.xml:
        logging.error("Invalid model path passed.")
        sys.exit("Please either pass a frozen pb or an Openvino Intermediate Representation xml model")

    if args.model_path:
        xml_path = optimize_network(args.model_path, source_fw="tf", network="tmp_net", input_shape=None, input_node="annette_bench1",
                         save_folder=args.save_folder)
    if  os.path.isfile(xml_path):
        run_network(xml_path=xml_path, report_dir="./tmp", hardware=args.device, batch=1, nireq=1, niter=10, api="async")

    logging.info("\n**********OPENVINO DONE**********")

if __name__ == "__main__":
    main()
