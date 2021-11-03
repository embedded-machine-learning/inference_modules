# this script takes a neural network in the TF Lite .tflite file,
# where all operations are supported by the EdgeTPU compiler,
# compiles it using the edgetpu_compiler
# and runs inference on the generated model

# Example: python3 inference.py --model_path ../../tests/networks/annette_bench1.tflite --save_folder ./tmp --device EDGETPU

import logging, argparse
import os, sys, time
import tflite_runtime.interpreter as tflite
import numpy as np

__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

def optimize_network(model_path="./models/model.tflite", save_folder = "./tmp"):
    logging.info("\nCompiling:", model_path)

    # check if necessary files exists
    if not os.path.isfile(model_path):
        logging.info("Model Optimizer not found at:", model_path)
        return False

    # check if save folder exists, create if it doesn't
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # compile network with edgetpu_compiler
    logging.info("\n**********EDGE TPU COMPILATION**********")
    edgetpu_compile_command = "edgetpu_compiler" + " --out_dir " + save_folder + " --show_operations " + model_path

    if os.system(edgetpu_compile_command):
        logging.info("\nAn error has occured during conversion!\n")
        return False

    edge_model_path = os.path.join(os.getcwd(), save_folder, model_path.split(".tflite")[0].split("/")[-1] + "_edgetpu.tflite")

    return edge_model_path



def run_network(edge_model_path="./tmp/model_edgetpu.tflite", data_dir="./tmp", niter = 10):
    # run inference with the edgetpu compiled model
    # invoke
    interpreter = tflite.Interpreter(
        model_path=edge_model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()

    # define input
    input_shape = input_details[0]['shape']
    input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)

    # invoke interpreter
    for i in range(niter):
        # print("invoking inference:", i)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()  # start inference
        time.sleep(0.01) # pause for 10ms between inference executions


def main():
    parser = argparse.ArgumentParser(description='Edge TPU Hardware Module')
    parser.add_argument("-m", '--model_path', default='./models/model.tflite',
                        help='Tflite model with Edge TPU supported operations', required=False)
    parser.add_argument("-sf", '--save_folder', default='./tmp',
                        help='folder to save the resulting files', required=False)
    args = parser.parse_args()

    edge_model_path = optimize_network(model_path=args.model_path, save_folder=args.save_folder)
    run_network(edge_model_path=edge_model_path, data_dir=args.save_folder, niter=10)


if __name__ == "__main__":
    main()