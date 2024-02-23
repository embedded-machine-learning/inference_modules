# this script calls the inference modules on a remote RPi4 ubuntu aarch64 server


# this script takes a neural network in the TFLite format
# And runs inference on the generated model using TFLite Interpereter, PyARMNN and a precompiled TFLite benchmark file
# Install Guide for Pyarmnn can be found here: https://developer.arm.com/documentation/102557/2108/Device-specific-installation/Raspberry-Pi-installation?lang=en
# Install Guide for TFLite Interpreter can be found here: https://www.tensorflow.org/lite/guide/python
# Performance measurement of TFLite https://www.tensorflow.org/lite/performance/measurement

# Example: python3 hw_modules/rpi4/inference.py --tflite_model ~/models/mobilenet_v1_1.0_224.tflite --save_dir tmp -n 10 -s 0 --bench_file linux_aarch64_benchmark_model --print --interpreter --pyarmnn

import logging
import os, fabric2
import yaml

from time import sleep, time
from datetime import datetime
from statistics import median
from yaml.loader import SafeLoader
from pathlib import Path

import numpy as np
import pandas as pd

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"



class rpi4Class:
    def __init__(self, config_file="../rpi4.yaml"):
        self.config_file = config_file
        f = open(os.path.abspath(config_file), "r")
        data = yaml.load(f, Loader=SafeLoader)
        f.close()
        self.ssh_ip = data['rpi4']['ssh_ip']
        self.ssh_key = data['rpi4']['ssh_key']
        self.ssh_user = data['rpi4']['ssh_user']
        self.port = data['rpi4']['port']
        self.tflite_model = data['rpi4']['tflite_model']
        self.model_path = data['rpi4']['model_path']
        self.save_dir = data['rpi4']['save_dir']
        self.niter = data['rpi4']['niter']
        self.threads = data['rpi4']['threads']
        self.bench_file = data['rpi4']['bench_file']
        self.ssh = data['rpi4']['ssh']
        self.print = data['rpi4']['print']
        self.interpreter = data['rpi4']['interpreter']
        self.pyarmnn = data['rpi4']['pyarmnn']
        self.sleep = data['rpi4']['sleep']



    def optimize_network(self, model_path="./models/model.pb", source_fw = "tflite", network = "tmp_net", input_shape = [1, 224, 224, 3],
                     input_node = "data", save_folder = "./tmp"):
        # TFLite does not support model conversion and the full Tensorflow cannot be installed on the RPi4
        print(network)
        return {"model_path": model_path, "tflite_model": network}





    def run_network_ssh(self, tflite_model = "model", model_path="./tmp/", save_dir = "./tmp", niter = 10, print_bool = False, sleep_time=0,
                    use_tflite=False, use_pyarmnn=False):
        
        print("run ssh")
        
        c = fabric2.Connection(host=self.ssh_ip, user=self.ssh_user, port=self.port, connect_kwargs={"key_filename": self.ssh_key})
        c.open()


        if not c.is_connected:
            print("ERROR: No SSH connection")
        else:
            print("*****in ssh*****")
            
            abs_folder = "/home/ubuntu/mwess/annette_ssh"

            c.run(f"mkdir -p {abs_folder}/inference/tmp")
            c.put(__file__, f"{abs_folder}/inference")
            c.put(self.config_file, abs_folder)


            c.put(str(Path(__file__).parent.absolute()) + "/linux_aarch64_benchmark_model", f"{abs_folder}/inference")
            c.put(str(model_path), "mwess/annette_ssh/inference")
            


            exec_command = f"source {abs_folder}/.venv_annette/bin/activate && " + f"cd {abs_folder}/inference && " + f"python3 inference.py --no-ssh --model_path {abs_folder}/inference/ --tflite_model " + tflite_model+".tflite --niter " + str(niter) + " --sleep " + str(sleep_time) + ""
            
            c.run(exec_command)
            

            c.get(f"{abs_folder}/inference/tmp/{tflite_model}_1thr.csv", save_dir + "/report.csv")

        c.close()

        print("rpi ssh done")

        return Path(save_dir, "report.csv")




    @staticmethod
    def run_network(self, tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 10, print_bool = False, sleep_time=0,
                    use_tflite=False, use_pyarmnn=False):
        # create report directory if it doesn't exist yet
        os.makedirs(save_dir, exist_ok=True)

        print("run Network method")
        print(use_tflite)


        # simple sanity check for sleep time
        if sleep_time and sleep_time > 10:
            print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(sleep_time))
            return
        elif sleep_time and sleep_time < 0:
            print("Invalid sleep time {0:.2f}s".format(sleep_time))
            return


        if use_tflite:
            print("run inference tflite")

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
            print("run inference pyarmnn")
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
            print("\nExecution time median: {:.3f} ms".format(median(times_times))) if print_bool else None


    def run_compiled_bench(self, tflite_path = "./tmp/model.tflite", save_dir = "./tmp", niter = 100, print_bool = False, 
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

    rpi4 = rpi4Class()


    import argparse
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 Inference Module')
    parser.add_argument("-tfl", '--tflite_model', default=rpi4.tflite_model, help='Name of TFLite model file', required=False)
    parser.add_argument("-mp", '--model_path', default=rpi4.model_path, help='Path to dir of model', required=False)
    parser.add_argument("-sd", '--save_dir', default=rpi4.save_dir, help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=rpi4.niter, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=rpi4.sleep, type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-thr", '--threads', default=rpi4.threads, type=int, help='number of threads to run compiled bench on', required=False)
    parser.add_argument("-bf", '--bench_file', default=rpi4.bench_file, type=str, help='path to compiled benchmark file', required=False)

    parser.add_argument('--ssh', dest='ssh', action='store_true', default=rpi4.ssh)
    parser.add_argument('--no-ssh', dest='ssh', action='store_false')

    parser.add_argument('--print', dest='print', action='store_true', default=rpi4.print)
    parser.add_argument('--no-print', dest='print', action='store_false')

    parser.add_argument('--interpreter', dest='interpreter', action='store_true', default=rpi4.interpreter)
    parser.add_argument('--no-interpreter', dest='interpreter', action='store_false')

    parser.add_argument('--pyarmnn', dest='pyarmnn', action='store_true', default=rpi4.pyarmnn)
    parser.add_argument('--no-pyarmnn', dest='pyarmnn', action='store_false')

    args = parser.parse_args()

    tflite_path = Path(args.model_path, args.tflite_model)

    print("in main tflite path", tflite_path)


    if not args.pyarmnn and not args.interpreter and not args.bench_file:
        logging.error("No Runtime chosen, please choose either PyARMNN, the TFLite Interpreter or provide a compiled benchmark file")
        return

    if args.ssh:
        rpi4.run_network_ssh(tflite_model=args.tflite_model, model_path=args.model_path, save_dir=args.save_dir, niter=args.niter,
                        print_bool=args.print, sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn)
        

    # if TFLite model is provided, use it for inference
    elif args.tflite_model and os.path.isfile(tflite_path):
        print("run tflite")
        rpi4.run_network(tflite_path=tflite_path, save_dir=args.save_dir, niter=args.niter,
                        print_bool=args.print, sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn)
    else:
        # if no neural network models are provided, return
        logging.error("Invalid model path {} passed.".format(args.tflite_model))
        return


    # run inference using the provided benchmark file if the benchmark file is valid
    if args.bench_file and os.path.isfile(args.bench_file) and not args.ssh:
        print("********Bench********")
        rpi4.run_compiled_bench(tflite_path=tflite_path, save_dir=args.save_dir, niter=args.niter, print_bool=args.print, 
        sleep_time=args.sleep, use_tflite=args.interpreter, use_pyarmnn=args.pyarmnn, bench_file=args.bench_file, num_threads = args.threads)

    logging.info("\n**********RPI INFERENCE DONE**********")


if __name__ == "__main__":
    main()



