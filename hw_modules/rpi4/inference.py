# this script takes a neural network in the Tensorflow format
# converts it to TFLite and runs inference on the generated model using TFLite Interpereter and PyARMNN

# Example: python3 inference.py --model_path TODO

import logging
import os, sys

from time import sleep, time
from datetime import datetime
from statistics import median

import numpy as np
import pyarmnn as ann
import tflite_runtime.interpreter as tflite


__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def optimize_network(model_path="./models/model.tflite", network = "tmp_net", input_shape = [1, 224, 224, 3], input_node = "data", save_folder = "./tmp"):
    # TFLite does not support model conversion and the full Tensorflow cannot be installed on the RPi4
    return model_path


def run_network(tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 10, print_bool = False, sleep_time=0,
                use_tflite=False, use_pyarmnn=False):
    # this function only makes sense if one framework is used
    if use_tflite:
        use_pyarmnn = False

    # create report directory if it doesn't exist yet
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    times = [] # list for latencies
    # simple sanity check for sleep time
    if sleep_time and sleep_time > 10:
        print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(sleep_time))
        return
    elif sleep_time and sleep_time < 0:
        print("Invalid sleep time {0:.2f}s".format(sleep_time))
        return

    if use_tflite:
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
        interpreter.invoke() # conduct warm up inference
    elif use_pyarmnn:
        parser = ann.ITfLiteParser()
        network = parser.CreateNetworkFromBinaryFile(tflite_path)

        options = ann.CreationOptions()
        runtime = ann.IRuntime(options)

        preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
        opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(),
                                             ann.OptimizerOptions())

        graph_id = 0
        input_names = parser.GetSubgraphInputTensorNames(graph_id)
        input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
        input_tensor_id = input_binding_info[0]
        input_tensor_info = input_binding_info[1]
        width, height = input_tensor_info.GetShape()[1], input_tensor_info.GetShape()[2]
        # print(f"tensor id: {input_tensor_id},tensor info: {input_tensor_info}")

        input_data = np.random.randint(0, 255, size=(height, width, 3))

        # Get output binding information for an output layer by using the layer name.
        output_names = parser.GetSubgraphOutputTensorNames(graph_id)
        output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
        output_tensors = ann.make_output_tensors([output_binding_info])

        net_id, _ = runtime.LoadNetwork(opt_network)
        if ann.TensorInfo.IsQuantized(input_tensor_info):
            input_data = np.uint8(input_data)
        else:
            input_data = np.float32(input_data/255)
        input_tensors = ann.make_input_tensors([input_binding_info], [input_data])

        runtime.EnqueueWorkload(0, input_tensors, output_tensors) # conduct warm up inference

    start_time = datetime.utcnow()
    try:
        for iteration in range(niter): # iterate over inferences
            inf_start_time = time()

            interpreter.invoke() if use_tflite else None # TFLite inference
            runtime.EnqueueWorkload(0, input_tensors, output_tensors) if use_pyarmnn else None # PyARMNN inference

            t_inf_ms = (time() - inf_start_time)*1000
            print("iteration {} took {:.3f} ms".format(iteration, t_inf_ms)) if print_bool else None
            times.append(t_inf_ms)
            sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")

    total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
    times.sort()
    print("\nExecution time median: {:.3f} ms".format(median(times))) if print_bool else None


def run_compiled_bench(tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 100, print_bool = False, 
sleep_time=0, use_tflite=False, use_pyarmnn=False, bench_file="./linux_aarch64_benchmark_model", num_threads = 1):
    # uses compiled benchmark by default so the use_* parameters have no effect
    use_tflite = False 
    use_pyarmnn = False
    sleep_time = 0

    print("Running benchmark with compiled app. Arguments sleep_time={}, use_tflite={} and \
    use_pyarmnn={} are ignored".format(sleep_time, use_tflite, use_pyarmnn))

    # create report directory if it doesn't exist yet
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # create name for profiler csv file generated by the benchmark, include number of threads into the name
    profile_csv_file_path = os.path.join(save_dir, tflite_path.split("/")[-1].split(".tflite")[0] + "_" + str(num_threads) + "thr" + ".csv")

    # invoke inference with linux_aarch64_benchmark_model
    invoke_command = " ".join(["./linux_aarch64_benchmark_model",
    "--graph=" + tflite_path,
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
    import argparse
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module')
    parser.add_argument("-tfl", '--tflite_model', help='TFLite model file', required=False)
    parser.add_argument("-sd", '--save_dir', default='./tmp', help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=0, type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-bf", '--bench_file', default="linux_aarch64_benchmark_model", type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--interpreter', dest='interpreter', action='store_true')
    parser.add_argument('--no-interpreter', dest='interpreter', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--pyarmnn', dest='pyarmnn', action='store_true')
    parser.add_argument('--no-pyarmnn', dest='pyarmnn', action='store_false')
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    if not args.pyarmnn and not args.interpreter and not args.bench_file:
        logging.error("No Runtime chosen, please choose either PyARMNN, the TFLite Interpreter or provide a compiled benchmark file")
        return

    # if TFLite model is provided, use it for inference
    if args.tflite_model and os.path.isfile(args.tflite_model):
        run_network(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter,
                        print_bool=args.print, sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn)
    else:
        # if no neural network models are provided, return
        logging.error("Invalid model path {} passed.".format(args.tflite_model))
        return

    # run inference using the provided benchmark file if the benchmark file is valid
    if args.bench_file and os.path.isfile(args.bench_file):
        run_compiled_bench(tflite_path=args.tflite_model, save_dir=args.save_dir, niter=args.niter, print_bool=args.print, 
        sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn, bench_file="./linux_aarch64_benchmark_model", num_threads = 1)

    logging.info("\n**********RPI INFERENCE DONE**********")

if __name__ == "__main__":
    main()
