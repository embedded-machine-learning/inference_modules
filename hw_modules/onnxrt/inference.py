
import logging
import os
import yaml

from time import sleep, time
from datetime import datetime
from statistics import median
from yaml.loader import SafeLoader
from pathlib import Path
import onnxruntime as onnxrt

import numpy as np
import subprocess

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


class onnxrtClass:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        f = open(os.path.abspath(config_file), "r")
        data = yaml.load(f, Loader=SafeLoader)
        f.close()
        hw = 'onnxrt'
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

    def optimize_network(self, model_path="./models/model.onnx",
                             source_fw="onnx", network="tmp_net",
                             input_shape=[1, 224, 224, 3], input_node="data",
                             save_folder="./tmp", **kwargs):
        logging.info("Optimize network method")

        return {"model_path": model_path,
                "save_dir": save_folder,
                "network": network}


    @staticmethod
    def run_network(model_path="", save_dir="./tmp", n = 100, **kwargs):
        # create report directory if it doesn't exist yet
        os.makedirs(save_dir, exist_ok=True)

        print("run Network method")
        sess_options = onnxrt.SessionOptions()
        sess_options.enable_profiling = False 
        sess = onnxrt.InferenceSession(model_path, sess_options)
        shape = sess.get_inputs()[0].shape
        input_name = sess.get_inputs()[0].name
        ximg = np.random.random_sample(shape)
        ximg = ximg.astype(np.float32)
        # warmup
        start = time()
        result = sess.run(None, {input_name: ximg})
        # inference
        for i in range(n-1):
            result = sess.run(None, {input_name: ximg})
        end = time()
        pf = sess.end_profiling()
        print(end-start)
        #print(pf)
        return pf

def main():

    onnxrtobj = onnxrtClass()

    import argparse
    parser = argparse.ArgumentParser(description='onnxrt Inference Module')
    parser.add_argument("-tfl", '--tflite_model', default=onnxrt.tflite_model,
                        help='Name of TFLite model file', required=False)
    parser.add_argument("-mp", '--model_path', default=onnxrt.model_path,
                        help='Path to dir of model', required=False)
    parser.add_argument("-sd", '--save_dir', default=onnxrt.save_dir,
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=onnxrt.niter,
                        type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=onnxrt.sleep, type=float,
                        help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-thr", '--threads', default=onnxrt.threads, type=int,
                        help='number of threads to run compiled bench on', required=False)
    parser.add_argument("-bf", '--bench_file', default=onnxrt.bench_file,
                        type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--ssh', dest='ssh',
                        action='store_true', default=onnxrt.ssh)
    parser.add_argument('--no-ssh', dest='ssh', action='store_false')

    parser.add_argument('--print', dest='print',
                        action='store_true', default=onnxrt.print)
    parser.add_argument('--no-print', dest='print', action='store_false')

    args = parser.parse_args()

    tflite_path = Path(args.model_path, args.tflite_model)

    print("in main tflite path", tflite_path)

    if args.ssh:
        onnxrt.run_network_ssh(tflite_model=args.tflite_model,
                               model_path=args.model_path,
                               save_dir=args.save_dir, niter=args.niter,
                               print_bool=args.print, sleep_time=args.sleep)

    """
    # if TFLite model is provided, use it for inference
    elif args.tflite_model and os.path.isfile(tflite_path):
        print("run tflite")
        onnxrt.run_network(tflite_path=tflite_path, save_dir=args.save_dir,
                        niter=args.niter, print_bool=args.print,
                        sleep_time=args.sleep)
    else:
        # if no neural network models are provided, return
        logging.error("Invalid model path {} passed.".format(args.tflite_model))
        return
    """

    # run inference using the provided benchmark file if the benchmark file is valid
    if args.bench_file and os.path.isfile(args.bench_file) and not args.ssh:
        print("********Bench********")
        onnxrt.run_compiled_bench(tflite_path=tflite_path,
                                  save_dir=args.save_dir,
                                  niter=args.niter,
                                  print_bool=args.print,
                                  sleep_time=args.sleep,
                                  bench_file=args.bench_file,
                                  num_threads=args.threads)

    logging.info("\n**********onnxrt INFERENCE DONE**********")


if __name__ == "__main__":
    main()
