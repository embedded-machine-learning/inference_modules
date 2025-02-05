import logging
import os
import yaml
from time import sleep, time
from datetime import datetime
from statistics import median
from yaml.loader import SafeLoader
from pathlib import Path

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


class xavierClass:
    def __init__(self, config_file="config.yaml", hw=None):
        # catch hw argument
        if hw is None:
            print("ERROR: Hardware not specified!")
            return
        # catch the case where the config file is not found
        if not os.path.isfile(config_file):
            print("ERROR: Config file not found!")
            return
        
        # load the config file
        self.config_file = config_file
        f = open(os.path.abspath(config_file), "r")
        data = yaml.load(f, Loader=SafeLoader)
        f.close()
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

    def optimize_network(self, model_path="./models/model.pb",
                             source_fw="onnx", network="tmp_net",
                             input_shape=[1, 224, 224, 3], input_node="data",
                             save_folder="./tmp", quant="int8"):
        logging.info("Optimize network method")
        quant_cmd = ""
        if quant == "int8":
            quant_cmd = " --int8"
        else:
            quant_cmd = " --fp16"

        import fabric2
        print(f"run ssh {self.ssh_key} {self.ssh_user} {self.ssh_ip}")
        c = fabric2.Connection(host=self.ssh_ip, user=self.ssh_user,
                               port=self.port, connect_kwargs={"key_filename": self.ssh_key})
        c.open()

        if not c.is_connected:
            print("ERROR: No SSH connection")
        else:
            print("*****in ssh*****")

            abs_folder = self.model_path
            print("model_path", model_path)
            model_path = Path(model_path)
            network = Path(network)
            print(self.config_file)

            c.run(f"mkdir -p {abs_folder}/inference/tmp")
            c.put(__file__, f"{abs_folder}/inference")
            c.put(self.config_file, f"{abs_folder}/inference")
            c.put(model_path, f"{abs_folder}/inference")
            # check if the model was uploaded
            c.run(f"ls -la {abs_folder}/inference")



            cmd = f"cd {abs_folder}/inference && /usr/src/tensorrt/bin/trtexec --onnx=./{model_path.name} {quant_cmd} --saveEngine=./tmp/{network.stem}.engine"
            c.run(cmd)
        # add .engine to model_path
        model_path = Path('tmp', network.stem + ".engine")
        print(model_path)
        c.close()
        print("xavier ssh done")

        return {"model_path": model_path,
                "save_dir": save_folder,
                "network": network}

    def run_network_ssh(self, model_path="./tmp/model.engine",
                        save_dir="./tmp", niter=10, print_bool=False,
                        sleep_time=0, network="network"):
        import fabric2
        from powerutils import measurement

        print("run ssh")

        c = fabric2.Connection(host=self.ssh_ip, user=self.ssh_user,
                               port=self.port, connect_kwargs={"key_filename": self.ssh_key})
        c.open()

        if not c.is_connected:
            print("ERROR: No SSH connection")
        else:
            print("*****in ssh*****")

            abs_folder = self.model_path
            #abs_folder = "/media/cdleml/512GB/Users/mwess/annette_ssh"

            exec_command = f"cd {abs_folder}/inference && /usr/src/tensorrt/bin/trtexec --loadEngine=./{model_path} --exportTimes={abs_folder}/inference/tmp/tmp.csv"
            c.run(exec_command)
            print(f"{abs_folder}/inference/tmp/tmp.csv")
            c.get(f"{abs_folder}/inference/tmp/tmp.csv",
                  str(save_dir) + "/xavier.csv")

        c.close()

        print("xavier ssh done")

        return Path(os.path.join(save_dir, "xavier.csv"))

    def run_network_ssh_perlayer(self, model_path="./tmp/model.engine",
                        save_dir="./tmp", niter=10, print_bool=False,
                        sleep_time=0, network="network", export="tmp"):
        import fabric2
        from powerutils import measurement

        print("run ssh")

        c = fabric2.Connection(host=self.ssh_ip, user=self.ssh_user,
                               port=self.port, connect_kwargs={"key_filename": self.ssh_key})
        c.open()

        if not c.is_connected:
            print("ERROR: No SSH connection")
        else:
            print("*****in ssh*****")

            abs_folder = self.model_path
            #abs_folder = "/media/cdleml/512GB/Users/mwess/annette_ssh"

            exec_command = f"cd {abs_folder}/inference && /usr/src/tensorrt/bin/trtexec --loadEngine=./{model_path} --exportTimes={abs_folder}/inference/tmp/tmp.csv --exportProfile={abs_folder}/inference/tmp/profile.csv"
            c.run(exec_command)
            print(f"{abs_folder}/inference/tmp/tmp.csv")
            c.get(f"{abs_folder}/inference/tmp/tmp.csv",
                  str(save_dir) + "/xavier.csv")
            c.get(f"{abs_folder}/inference/tmp/profile.csv",
                  str(save_dir) + "/xavier_profile.csv")

        c.close()

        print("xavier ssh done")

        return Path(os.path.join(save_dir, "xavier.csv"))

    @staticmethod
    def run_network(tflite_path="./tmp/model.tflite", save_dir="./tmp", niter=10, print_bool=False, sleep_time=0):
        # create report directory if it doesn't exist yet
        os.makedirs(save_dir, exist_ok=True)

        print("run Network method")

        # simple sanity check for sleep time
        if sleep_time and sleep_time > 10:
            print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(
                sleep_time))
            return
        elif sleep_time and sleep_time < 0:
            print("Invalid sleep time {0:.2f}s".format(sleep_time))
            return

        print("run inference tflite")

        import tflite_runtime.interpreter as tflite
        times = []  # list for latencies
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # print information about model inputs and outputs
        print("\n*****Model Inputs*****:")
        for input in input_details:
            for v, k in input.items():
                print(v, k) if print_bool else None
            print() if print_bool else None
        print() if print_bool else None

        print("\n*****Model Outputs*****")
        for output in output_details:
            for v, k in output.items():
                print(v, k) if print_bool else None
            print() if print_bool else None
        print() if print_bool else None

        # generate random input
        if input['dtype'] == np.uint8:
            input_data = np.random.randint(low=np.iinfo(input['dtype']).min, high=np.iinfo(input['dtype']).max,
                                           size=input_details[0]['shape'], dtype=input_details[0]["dtype"])
        elif input['dtype'] == np.float32:
            input_data = np.random.default_rng().standard_normal(size=input_details[0]['shape'],
                                                                 dtype=input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()  # conduct warm up inference
        start_time = datetime.utcnow()

        # run inference
        try:
            for iteration in range(niter):  # iterate over inferences
                inf_start_time = time()

                interpreter.invoke()  # TFLite inference

                t_inf_ms = (time() - inf_start_time)*1000
                print("iteration {} took {:.3f} ms".format(
                    iteration, t_inf_ms)) if print_bool else None
                times.append(t_inf_ms)
                sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        times.sort()
        print("\nExecution time median: {:.3f} ms".format(
            median(times))) if print_bool else None

    def run_compiled_bench(self, tflite_path="./tmp/model.tflite", save_dir="./tmp", niter=100, print_bool=False,
                           sleep_time=0, bench_file="./benchmark_model_xavier", num_threads=4):
        # uses compiled benchmark by default so the use_* parameters have no effect
        use_tflite = False
        use_pyarmnn = False
        sleep_time = 0

        print("Running benchmark with compiled app. Arguments sleep_time={}, use_tflite={} and \
        use_pyarmnn={} are ignored".format(sleep_time, use_tflite, use_pyarmnn))

        # create report directory if it doesn't exist yet
        os.makedirs(save_dir, exist_ok=True)

        # just get filename from path
        tflite_file = os.path.split(tflite_path)[1]
        print(tflite_file)

        # create name for profiler csv file generated by the benchmark, include number of threads into the name
        # profile_csv_file_path = os.path.join(save_dir, os.path.splitext(os.path.split(tflite_path)[1])[0] + "_" + str(num_threads) + "thr" + ".csv")
        profile_csv_file_path = os.path.join(save_dir, "tmp.csv")
        print(profile_csv_file_path)

        # invoke inference with linux_aarch64_benchmark_model
        invoke_command = " ".join([bench_file,
                                   "--graph=" + str(tflite_file),
                                   "--num_threads=" + str(num_threads),
                                   "--num_runs=" + str(niter),
                                   "--verbose=true",
                                   "--enable_op_profiling=true",
                                   "--profiling_output_csv_file=" + profile_csv_file_path])

        if os.system(invoke_command):
            print("\nSomething went wrong during compiled benchmark execution\n")
            return

        print("\nBenchmark using a compiled bench has been completed\n")


def main():

    xavier = xavierClass()

    import argparse
    parser = argparse.ArgumentParser(description='xavier Inference Module')
    parser.add_argument("-tfl", '--tflite_model', default=xavier.tflite_model,
                        help='Name of TFLite model file', required=False)
    parser.add_argument("-mp", '--model_path', default=xavier.model_path,
                        help='Path to dir of model', required=False)
    parser.add_argument("-sd", '--save_dir', default=xavier.save_dir,
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=xavier.niter,
                        type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=xavier.sleep, type=float,
                        help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-thr", '--threads', default=xavier.threads, type=int,
                        help='number of threads to run compiled bench on', required=False)
    parser.add_argument("-bf", '--bench_file', default=xavier.bench_file,
                        type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--ssh', dest='ssh',
                        action='store_true', default=xavier.ssh)
    parser.add_argument('--no-ssh', dest='ssh', action='store_false')

    parser.add_argument('--print', dest='print',
                        action='store_true', default=xavier.print)
    parser.add_argument('--no-print', dest='print', action='store_false')

    args = parser.parse_args()

    tflite_path = Path(args.model_path, args.tflite_model)

    print("in main tflite path", tflite_path)

    if args.ssh:
        xavier.run_network_ssh(tflite_model=args.tflite_model,
                               model_path=args.model_path,
                               save_dir=args.save_dir, niter=args.niter,
                               print_bool=args.print, sleep_time=args.sleep)

    """
    # if TFLite model is provided, use it for inference
    elif args.tflite_model and os.path.isfile(tflite_path):
        print("run tflite")
        xavier.run_network(tflite_path=tflite_path, save_dir=args.save_dir,
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
        xavier.run_compiled_bench(tflite_path=tflite_path,
                                  save_dir=args.save_dir,
                                  niter=args.niter,
                                  print_bool=args.print,
                                  sleep_time=args.sleep,
                                  bench_file=args.bench_file,
                                  num_threads=args.threads)

    logging.info("\n**********xavier INFERENCE DONE**********")


if __name__ == "__main__":
    main()
