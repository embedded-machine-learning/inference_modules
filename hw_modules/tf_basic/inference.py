import sys, time, logging
import os
import numpy as np
import tensorflow.compat.v1 as tf
import yaml

from time import sleep, time
from datetime import datetime
from statistics import median
from yaml.loader import SafeLoader
from pathlib import Path
from pathlib import Path

from tensorflow.python.client import timeline
tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()

class tf_basicClass:
    def __init__(self, config_file="../tfbasic.yaml"):
        self.config_file = config_file
        f = open(os.path.abspath(config_file), "r")
        data = yaml.load(f, Loader=SafeLoader)
        f.close()
        name = 'tf_basic'
        self.ssh_ip = data[name]['ssh_ip']
        self.ssh_key = data[name]['ssh_key']
        self.ssh_user = data[name]['ssh_user']
        self.port = data[name]['port']
        self.tflite_model = data[name]['tflite_model']
        self.model_path = data[name]['model_path']
        self.save_dir = data[name]['save_dir']
        self.niter = data[name]['niter']
        self.threads = data[name]['threads']
        self.bench_file = data[name]['bench_file']
        self.ssh = data[name]['ssh']
        self.print = data[name]['print']
        self.interpreter = data[name]['interpreter']
        self.pyarmnn = data[name]['pyarmnn']
        self.sleep = data[name]['sleep']


    def optimize_network(self, model_path = "", source_fw = "tf", network = "tmp_net", input_shape = [1, 224, 224, 3],
                          input_node = "Placeholder", save_folder = Path("./tmp")):
        return {"model_path": model_path, "source_fw": source_fw, "network": network, "input_shape": input_shape, "input_node": input_node}

    def run_network(self, model_path = "", network = "tmp_net", input_shape = [1, 224, 224, 3] ,
                    save_folder = Path("./database/benchmarks/tmp/"), **kwargs):
        model_filepath = str(model_path)

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        graph = wrapped_import.graph

        out_nodes = []
        in_nodes = []
        for n in graph_def.node:
            if len(n.input) > 0:
                in_nodes.extend(n.input)
            out_nodes.append(n.name)

        """remove all nodes that are in input list from output list"""
        out_nodes = [n for n in out_nodes if n not in in_nodes]
        logging.debug(f"Output nodes: {out_nodes}")
        
        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        inputs = [n.name for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)
        in_shape = []
        for (n, node) in enumerate(graph_def.node):
            if n == 0:
                for a in node.attr['shape'].shape.dim:
                    in_shape.append(int(a.size))

        graph.finalize()

        print(inputs)

        data = np.ones(in_shape)
        sess = tf.Session(graph = graph)

        print("Run TF Profiler")

        with tf.Session(graph = graph) as sess:
            # add additional options to trace the session execution
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            input = graph.get_tensor_by_name(f"{inputs[0]}:0")
            output_tensor = graph.get_tensor_by_name(f"{out_nodes[0]}:0")
            output = sess.run(output_tensor, feed_dict = {input: data}, options=options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()

            # create a folder to save the timeline file
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            out = f'{str(save_folder)}/{network}.json'
            with open(out, 'w') as f:
                f.write(chrome_trace)
            return out

        start_time = time.clock()
        output = sess.run(output_tensor, feed_dict = {input: data})
        end_time = time.clock()
        print(end_time - start_time, "seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NCS2 power benchmark')
    parser.add_argument("-p", '--pb', default='yolov3.pb',
                        help='intermediade representation', required=False)
    parser.add_argument("-x", '--xml', default='yolov3.xml',
                        help='movidius representation', required=False)
    parser.add_argument("-sf", '--save_folder', default='save',
                        help='folder to save the resulting files', required=False)
    parser.add_argument("-a", '--api', default='sync',
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

    if not args.pb and not args.xml:
        sys.exit("Please pass either a frozen pb or IR xml/bin model")
    
    # TODO reattach the converter inference etc
