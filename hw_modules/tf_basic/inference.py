import sys, time, logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


def optimize_network(pb, source_fw = "tf", network = "tmp_net", image = [1, 224, 224, 3] , input_node = "data", save_folder = "./tmp"):
    return True 


def run_network(xml_path = None, report_dir = "./tmp", hardware = "MYRIAD", batch = 1, nireq = 1, niter = 10, api = "sync"):
    return True


def run_inference(pb, source_fw = "tf", network = "tmp_net", image = [1, 224, 224, 3] , input_node = "Placeholder", output_node = "flatten_Reshape", save_folder = "./database/benchmarks/tmp/"):
    model_filepath = str(pb)

    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    graph = wrapped_import.graph

    print('Check out the input placeholders:')
    nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)
    in_shape = []
    for (n, node) in enumerate(graph_def.node):
        if n == 0:
            for a in node.attr['shape'].shape.dim:
                in_shape.append(int(a.size))

    graph.finalize()

    data = np.ones(in_shape)
    sess = tf.Session(graph = graph)

    print("Run TF Profiler")
    with tf.Session(graph = graph) as sess:
        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        input = graph.get_tensor_by_name("Placeholder:0")
        output_tensor = graph.get_tensor_by_name("flatten_Reshape:0")
        output = sess.run(output_tensor, feed_dict = {input: data}, options=options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        print(save_folder)
        with open(str(save_folder)+'/timeline_01.json', 'w') as f:
            f.write(chrome_trace)

    #start_time = time.clock()
    #output = sess.run(output_tensor, feed_dict = {input: data})
    #end_time = time.clock()
    #print(end_time - start_time, "seconds")


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
