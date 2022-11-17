# this script takes a neural network in the TFLite format
# And runs inference on the generated model using TFLite Interpereter, PyARMNN and a precompiled TFLite benchmark file
# Install Guide for Pyarmnn can be found here: https://developer.arm.com/documentation/102557/2108/Device-specific-installation/Raspberry-Pi-installation?lang=en
# Install Guide for TFLite Interpreter can be found here: https://www.tensorflow.org/lite/guide/python
# Performance measurement of TFLite https://www.tensorflow.org/lite/performance/measurement

# Example: python3 hw_modules/rpi4/inference.py --tflite_model ~/models/mobilenet_v1_1.0_224.tflite --save_dir tmp -n 10 -s 0 --bench_file linux_aarch64_benchmark_model --print --interpreter --pyarmnn

import logging, json
import os, sys

from time import sleep, time
from datetime import datetime
from statistics import median

import numpy as np
import pandas as pd

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def optimize_network(model_path="./models/model.tflite", network = "tmp_net", input_shape = [1, 224, 224, 3], input_node = "data", save_folder = "./tmp"):
    # TFLite does not support model conversion and the full Tensorflow cannot be installed on the RPi4
    return model_path


def run_network(tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 10, print_bool = False, sleep_time=0,
                use_tflite=False, use_pyarmnn=False):
    # create report directory if it doesn't exist yet
    os.makedirs(save_dir, exist_ok=True)

    # simple sanity check for sleep time
    if sleep_time and sleep_time > 10:
        print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(sleep_time))
        return
    elif sleep_time and sleep_time < 0:
        print("Invalid sleep time {0:.2f}s".format(sleep_time))
        return

    if use_tflite:
        import tflite_runtime.interpreter as tflite
        times = [] # list for latencies
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
        start_time = datetime.utcnow()

        # run inference
        try:
            for iteration in range(niter): # iterate over inferences
                inf_start_time = time()

                interpreter.invoke() # TFLite inference

                t_inf_ms = (time() - inf_start_time)*1000
                print("iteration {} took {:.3f} ms".format(iteration, t_inf_ms)) if print_bool else None
                times.append(t_inf_ms)
                sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        times.sort()
        print("\nExecution time median: {:.3f} ms".format(median(times))) if print_bool else None

    if use_pyarmnn:
        import pyarmnn as ann
        times_times = [] # list for latencies
        parser = ann.ITfLiteParser()
        network = parser.CreateNetworkFromBinaryFile(tflite_path)

        options = ann.CreationOptions()
        runtime = ann.IRuntime(options)
        
        print("Supported Backends:",runtime.GetDeviceSpec())
        # CpuAcc is usually much faster than CpuRef and should be the first choice
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
        
        # enable profiler as in https://github.com/ARM-software/armnn/issues/468
        profiler = runtime.GetProfiler(graph_id)
        profiler.EnableProfiling(True)

        runtime.EnqueueWorkload(net_id, input_tensors, output_tensors) # conduct warm up inference
        start_time = datetime.utcnow()

        profile_results = {} # empty dict to contain profiler results for every inference for every layer

        # run inference
        try:
            for iteration in range(niter): # iterate over inferences
                inf_start_time = time()

                runtime.EnqueueWorkload(0, input_tensors, output_tensors) # PyARMNN inference

                t_inf_ms = (time() - inf_start_time)*1000
                #print("iteration {} took {:.3f} ms".format(iteration, t_inf_ms)) if print_bool else None
                times_times.append(t_inf_ms)
                sleep(sleep_time)

                profiler_data = ann.get_profiling_data(profiler)
                # Calculate mean of all total inference times (convert from us to ms).
                total_inference_times = profiler_data.inference_data['execution_time']
                mean_total_time = np.mean(total_inference_times) / 1000
                #print("Total inference time (ms): {}".format(mean_total_time))
                # Calculate mean of all inferences by layer (convert from us to ms).

                layer_time_backends = []
                for key, val in profiler_data.per_workload_execution_data.items():
                    mean_layer_time = np.mean(val['execution_time']) / 1000 # val execution time contains two values in us
                    layer_time_backends.append((key, mean_layer_time, val['backend'])) # add layer name, execution time and backend
                    
                    #print("Layer: {}, Time (ms): {}, Backend: {}".format(key, mean_layer_time, val['backend']))

                    # in the first iteration, the dictionary is initialized with keys
                    if iteration == 0:
                        profile_results[key] = [mean_layer_time]
                    else:
                        # in all other iterations, the layer time is added to the list for the initialized layer name
                        profile_results[key] += [mean_layer_time]

                cpuref_time = 0
                cpuacc_time = 0
                other_time = 0
                for layer_time_backend in layer_time_backends:
                    _, time_2, backend = layer_time_backend
                    if backend == "CpuRef":
                        cpuref_time += time_2
                    elif backend == "CpuAcc":
                        cpuacc_time += time_2
                    else:
                        other_time += time_2
                print("CpuRef time (ms): {:.3f}, CpuAcc time (ms) {:.3f}, Other time (ms): {:.3f}".format(cpuref_time, cpuacc_time, other_time))
        except KeyboardInterrupt:
            print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")
        
        # calculate the layer average of all inference iterations and gather them into one list
        layer_avg_time_list = []
        for key,val in profile_results.items():
            layer_time_avg = np.average(val)
            print("{}: {:.3f} ms".format(key, layer_time_avg))
            layer_avg_time_list.append((key, layer_time_avg))

        # make sure a separate directory for pyarmnn json files exists
        pyarmnn_profiler_dir = "pyarmnn_json"
        os.makedirs(os.path.join(save_dir, pyarmnn_profiler_dir),  exist_ok=True)

        # construct the json file name from the tflite model name
        pandas_layer_time = pd.DataFrame(layer_avg_time_list, columns=["Layer", "latency [ms]"])
        outfile = os.path.join(save_dir, pyarmnn_profiler_dir, os.path.splitext(os.path.split(tflite_path)[1])[0] + ".json")
        # save PyARMNN profiler results to file
        _ = pandas_layer_time.to_json(outfile, orient='index')

    total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
    times_times.sort()
    #print("\nExecution time median: {:.3f} ms".format(median(times_times))) if print_bool else None


def run_compiled_bench(tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 100, print_bool = False, 
sleep_time=0, use_tflite=False, use_pyarmnn=False, bench_file="./linux_aarch64_benchmark_model", num_threads = 4):
    # uses compiled benchmark by default so the use_* parameters have no effect
    use_tflite = False 
    use_pyarmnn = False
    sleep_time = 0

    print("Running benchmark with compiled app. Arguments sleep_time={}, use_tflite={} and \
    use_pyarmnn={} are ignored".format(sleep_time, use_tflite, use_pyarmnn))

    # create report directory if it doesn't exist yet
    os.makedirs(save_dir, exist_ok=True)

    # create name for profiler csv file generated by the benchmark, include number of threads into the name
    profile_csv_file_path = os.path.join(save_dir, os.path.splitext(os.path.split(tflite_path)[1])[0] + "_" + str(num_threads) + "thr" + ".csv")

    # invoke inference with linux_aarch64_benchmark_model
    invoke_command = " ".join([bench_file,
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
    parser.add_argument("-thr", '--threads', default=1, type=int, help='number of threads to run compiled bench on', required=False)
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
        sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn, bench_file=args.bench_file, num_threads = args.threads)

    logging.info("\n**********RPI INFERENCE DONE**********")

if __name__ == "__main__":
    main()
