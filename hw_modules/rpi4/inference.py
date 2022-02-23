# this script takes a neural network in the Tensorflow format
# converts it to TFLite and runs inference on the generated model using TFLite Interpereter and PyARMNN

# Example: python3 inference.py --model_path TODO

import logging
import os, sys
from time import sleep, time
from datetime import datetime


__author__ = "Matvey Ivanov"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"


def optimize_network(model_path="./models/model.pb", network = "tmp_net", input_shape = [1, 224, 224, 3] , input_node = "data", save_folder = "./tmp"):
    logging.info("\n**********TFLITE CONVERSION**********")
    # check if model path valid
    if os.path.isfile(model_path):
        # Tensorflow LITE conversion
        # TODO implement model conversion
        tflite_path = os.path.join(save_folder, "model")
        logging.debug(tflite_path)
    else:
        logging.info("Invalid Model Path: {} given!".format(model_path))

    logging.info("\n**********TFLITE MODEL GENERATED AT {}**********".format(tflite_path))
    # return path to generated TFLite model
    return tflite_path


def run_network(tflite_path = "./tmp/model.tflite", report_dir = "./tmp", niter = 10,
                print_bool = False, sleep_time=0, power_meas=False):
    # initialize power measurement
    if power_meas:
        pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=0)
        model_name_kwargs = {"model_name": "test", "custom_param": "infmod"}

    # create report directory if it doesn't exist yet
    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    # start inference over niter

    times = [] # list for latency
    # simple sanity check for sleep time
    if sleep_time and sleep_time > 10:
        print("Time between iterations was set to {0:.2f}s. Please choose a float < 10".format(sleep_time))
        return
    elif sleep_time and sleep_time < 0:
        print("Invalid sleep time {0:.2f}s".format(sleep_time))
        return

    start_time = datetime.utcnow()
    if power_meas:
        pm.start_gather(model_name_kwargs)  # start power measurement

    try:
        for iteration in range(niter): # iterate over inferences
            # TODO implement inference
            sleep(0.01)
            print(iteration, "inference conducted")
            if print_bool:
                print("iteration {} took {:.3f} ms".format(iteration, iteration))
            times.append(iteration)
            sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nInference loop exited via KeyboardInterrupt (ctrl + c)")

    if power_meas:
        pm.end_gather(True) # end powermeasurement
        print("Power Measurement ended")

    total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
    times.sort()
    if print_bool:
        print("Execution time median: {:.3f} ms".format(median(times)))

    # TODO save profiler data if available


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Raspberry Pi 4 power benchmark')
    parser.add_argument("-tf", '--tf_model', help='Tensorflow model file', required=False)
    parser.add_argument("-tfl", '--tflite_model', help='TFLite model file', required=False)
    parser.add_argument("-sf", '--save_folder', default='./tmp', help='folder to save the resulting files', required=False)
    parser.add_argument("-n", '--niter', default=10, type=int, help='number of iterations', required=False)
    parser.add_argument("-s", '--sleep', default=0, type=float, help='time to sleep between inferences in seconds', required=False)
    parser.add_argument("-rd", '--report_dir', default='reports', help='Directory to save reports into', required=False)

    parser.add_argument('--print', dest='print', action='store_true')
    parser.add_argument('--no-print', dest='print', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('--interpreter', dest='interpreter', action='store_true')
    parser.add_argument('--no-interpreter', dest='interpreter', action='store_false')
    parser.set_defaults(feature=True)

    parser.add_argument('--pyarmnn', dest='pyarmnn', action='store_true')
    parser.add_argument('--no-pyarmnn', dest='pyarmnn', action='store_false')
    parser.set_defaults(feature=True)

    parser.add_argument('--power_meas', dest='power_meas', action='store_true')
    parser.add_argument('--no-power_meas', dest='power_meas', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()
    tflite_path = args.tflite_model

    # if no neural network models are provided, exit script
    if not args.tf_model and not args.tflite_model:
        logging.error("Invalid model path passed.")
        sys.exit("Please provide with either a Tensorflow or TFLite model file")

    # invoke model conversion if only Tensorflow model file is provided
    if args.tf_model and not args.tflite_model:
        # overwrite tflite_path during network optimization
        tflite_path = optimize_network(args.tf_model, network="tmp_net", input_shape=[1,416,416,3],
                                    input_node="annette_bench1", save_folder=args.save_folder)

    # only do power measurements if flag is set
    if args.power_meas:
        from powerutils import measurement

    # if TFLite model is provided, use it for inference and ignore tf_model
    if args.tflite_model and os.path.isfile(args.tflite_model):
        run_network(tflite_path=tflite_path, report_dir="./tmp", niter=args.niter,
                        print_bool=args.print, sleep_time=args.sleep, power_meas=args.power_meas)

    logging.info("\n**********OPENVINO DONE**********")

if __name__ == "__main__":
    main()
