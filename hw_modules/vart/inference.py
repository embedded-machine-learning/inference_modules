import subprocess
import os
import numpy as np
from PIL import Image
from numpy.lib.function_base import copy
import paramiko
import time
import sys
from powerutils import measurement
import argparse


input_fn_file = "./json_files/my_input_fn.py"

fn_file = ['import numpy as np\n', 'import PIL\n', 'from PIL import Image\n', 'import os\n', '\n', "input_node = 'input'\n", "output_node = 'resnet_v1_50/predictions/Reshape_1'\n", '\n', "image_list = os.listdir('./calib')\n", '\n', 'def calib_input(iter):\n', "    image = Image.open('./calib/'+image_list[iter])\n", '    array = np.asarray(image)\n', '    array = array[np.newaxis, ...]\n', '    return {input_node:array}\n', '\n']
trace_json = ['{\n', '\t"options": {\n', '\t\t"runmode": "debug",\n', '\t\t"cmd": "/home/root/workspace/inferenz.py 1 /home/root/workspace/compiled_model.xmodel",\n', '\t\t"output": "./trace.xsat",\n', '\t\t"timeout": 5\n', '\t},\n', '\t"trace": {\n', '\t\t"enable_trace_list": ["vitis-ai-library", "vart", "costum"],\n', '\t\t"trace_costum": [ "TopK", "CPUCalcSoftmax"]\n', '\t}\n', '}\n', '\n']
start_docker_command = ['docker', 'run', '--device=/dev/dri/renderD128', '-v', '/dev/shm:/dev/shm', '-v', '/opt/xilinx/dsa:/opt/xilinx/dsa', '-v', '/opt/xilinx/overlaybins:/opt/xilinx/overlaybins', '-v', '/etc/xbutler:/etc/xbutler', '-e', 'USER=intel-nuc', '-e', 'UID=1000', '-e', 'GID=1000', '-e', 'VERSION=latest', '-v', '/home/intel-nuc/Vitis-AI:/vitis_ai_home', '-v', '/home/intel-nuc/marco/inference-modules/hw_modules/vart:/workspace', '-w', '/workspace', '--rm', '--network=host', '-d', '-it', 'xilinx/vitis-ai-cpu', 'bash']
docker_methods = ['import argparse\n', 'import os\n', 'import tensorflow as tf\n', 'from tensorflow.python.platform import gfile\n', 'import time\n', '\n', 'input_fn_file = "./my_input_fn.py"\n', 'zcu102_image = "Standard_image.json"\n', '\n', '\n', 'def write_lines(replace_string, search_string):\n', '\n', '    with open(input_fn_file, "r") as f:\n', '        searchlines = f.readlines()\n', '    for i, line in enumerate(searchlines):\n', '        if search_string in line:\n', '            searchlines[i] = replace_string \n', '\n', '    with open(input_fn_file, "w") as f:\n', '        f.writelines(searchlines)\n', '        f.close\n', '    return\n', '\n', '\n', '\n', 'def get_input_output_layer(path):\n', '    print("get input output layer")\n', '    GRAPH_PB_PATH = path\n', '    with tf.Session() as sess:\n', '\n', "        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:\n", '            graph_def = tf.GraphDef()\n', '    graph_def.ParseFromString(f.read())\n', '    sess.graph.as_default()\n', "    #tf.import_graph_def(graph_def, name='')\n", '    graph_nodes=[n for n in graph_def.node]\n', '    names = []\n', '    for t in graph_nodes:\n', '        names.append(t.name)\n', '\n', '    input_node = names[0]\n', '    output_node = names[-1]\n', '\n', '    node_string = f"input_node = \'{input_node}\'\\n"\n', '    write_lines(node_string,"input_node =")\n', '    node_string = f"output_node = \'{output_node}\'\\n"\n', '    write_lines(node_string,"output_node =")\n', '\n', '\n', '    return()\n', '\n', 'def quantize(**kwargs):\n', '\n', '    print("quantize\\n")\n', '    x =  os.getcwd()\n', "    x = os.listdir('./')\n", '    cmd = f"vai_q_tensorflow quantize --input_frozen_graph \'{kwargs[\'pb\']}\' --input_nodes \'{kwargs[\'input_node\']}\' --input_shapes \'{kwargs[\'batch\']}\',\'{kwargs[\'height\']}\',\'{kwargs[\'width\']}\',\'{kwargs[\'channels\']}\' --output_nodes \'{kwargs[\'output_node\']}\' --output_dir vai_q_output/\'{kwargs[\'name\']}\' --calib_iter 1 --input_fn my_input_fn.calib_input"\n', '    print(cmd)\n', '\n', '    time.sleep(5)\n', '    os.system(cmd)\n', '\n', '\n', '    return\n', '\n', ' \n', '\n', '\n', '\n', 'def str2list(v):\n', '    #print("str2list")\n', '    #print(v)\n', "    v = v.replace('[','')\n", "    v = v.replace(']','')\n", '    v = v.replace("\'",\'\')\n', "    liste = v.split(',')\n", '    #print(liste)\n', '    batch = int(liste[0])\n', '    w = int(liste[1])\n', '    h = int(liste[2])\n', '    c = int(liste[3])\n', '    #print(liste)\n', '    return([batch,w,h,c])\n', '\n', 'def str2bool(v):\n', '    if isinstance(v, bool):\n', '        return v\n', "    if v.lower() in ('yes', 'true', 't', 'y', '1'):\n", '        return True\n', "    elif v.lower() in ('no', 'false', 'f', 'n', '0'):\n", '        return False\n', '    else:\n', "        raise argparse.ArgumentTypeError('Boolean value expected.')\n", '\n', '\n', '\n', 'def main():\n', "    parser = argparse.ArgumentParser(description='Xilinx inference benchmark')\n", '    parser.add_argument("-p", "--pb", default="./models/annete/annette_bench1.pb")\n', "    parser.add_argument('-sf', '--save-folder', default='./tmp')\n", "    parser.add_argument('-fw', '--framework', default='tf')\n", "    #parser.add_argument('-i', '--image', type=str2list, default=[1,1,1,3])\n", "    parser.add_argument('-d','--docker', type=str2bool,default =False)\n", "    parser.add_argument('-m','--name', default='network')\n", "    parser.add_argument('-e','--execute', default='quantize')\n", "    parser.add_argument('-x','--height', default= 1)\n", "    parser.add_argument('-w', '--width', default = 1)\n", "    parser.add_argument('-c', '--channels', default = 3)\n", "    parser.add_argument('-b', '--batch' , default = 1)\n", "    parser.add_argument('-i', '--input_node', default = 'x')\n", "    parser.add_argument('-o', '--output_node', default = 'x')\n", '    args = parser.parse_args()\n', '\n', '    dictonary = vars(args)\n', '    print(dictonary)\n', '\n', '\n', '    print(args)\n', '\n', '\n', "    if(args.execute=='get_nodes'):\n", '        print("insert get_nodes methode")\n', '        get_input_output_layer(args.pb)\n', "    if(args.execute=='quantize'):\n", '        quantize(**dictonary)\n', '\n', '\n', 'if __name__=="__main__":\n', '    main()\n', '\n']
inferenz_base =  ['from ctypes import *\n', 'from typing import List\n', 'import cv2\n', 'import numpy as np\n', 'import xir\n', 'import vart\n', 'import os\n', 'import math\n', 'import threading\n', 'import time\n', 'import sys\n', '\n', 'from vaitrace_py import vai_tracepoint\n', '\n', '\n', 'width = xxx\n', 'height = xxx\n', '\n', '\n', '\n', '_B_MEAN = 104.0\n', '_G_MEAN = 107.0\n', '_R_MEAN = 123.0\n', 'MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]\n', 'SCALES = [1.0, 1.0, 1.0]\n', '\n', 'def preprocess_one_image_fn(image_path, width=width, height=height):\n', '   means = MEANS\n', '   scales = SCALES\n', '   image = cv2.imread(image_path)\n', '   image = cv2.resize(image,(width, height))\n', '   B, G, R = cv2.split(image)\n', '   B = (B - means[0]) * scales[0]\n', '   G = (G - means[1]) * scales[1]\n', '   R = (R - means[2]) * scales[2]\n', '   return image\n', '\n', '\n', 'calib_image_dir = "/home/root/workspace/images/"\n', 'global threadnum\n', 'threadnum = 0\n', '\n', '\n', 'def runNet(runner: "Runner", img, cnt):\n', '    """get tensor"""\n', '    inputTensors = runner.get_input_tensors()\n', '    outputTensors = runner.get_output_tensors()\n', '    input_ndim = tuple(inputTensors[0].dims)\n', '    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])\n', '\n', '    output_ndim = tuple(outputTensors[0].dims)\n', '    n_of_images = len(img)\n', '    count = 0\n', '    while count < cnt:\n', '        runSize = input_ndim[0]\n', '        """prepare batch input/output """\n', '        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]\n', '        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]\n', '\n', '        """init input image to input buffer """\n', '        for j in range(runSize):\n', '            imageRun = inputData[0]\n', '            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])\n', '        starttime = time.time()\n', '        """run with batch """\n', '        job_id = runner.execute_async(inputData, outputData)\n', '        runner.wait(job_id)\n', '        timetotal = time.time() - starttime\n', "        timefile = open('./time.txt','a')\n", '        timefile.write(str(timetotal)+"\\n")\n', '        timefile.close()\n', '\n', '        count = count + runSize\n', '"""\n', ' obtain dpu subgrah\n', '"""\n', 'def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:\n', '    assert graph is not None, "\'graph\' should not be None."\n', '    root_subgraph = graph.get_root_subgraph()\n', '    assert (\n', '        root_subgraph is not None\n', '    ), "Failed to get root subgraph of input Graph object."\n', '    if root_subgraph.is_leaf:\n', '        return []\n', '    child_subgraphs = root_subgraph.toposort_child_subgraph()\n', '    assert child_subgraphs is not None and len(child_subgraphs) > 0\n', '    return [\n', '        cs\n', '        for cs in child_subgraphs\n', '        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"\n', '    ]\n', '\n', '\n', 'def main(argv):\n', '    global threadnum\n', '    listimage = os.listdir(calib_image_dir)\n', '    threadAll = []\n', '    threadnum = int(argv[1])\n', '    i = 0\n', '    global runTotall\n', '    runTotall = len(listimage)\n', '    g = xir.Graph.deserialize(argv[2])\n', '    subgraphs = get_child_subgraph_dpu(g)\n', '    #assert len(subgraphs) == 1  # only one DPU kernel\n', '    all_dpu_runners = []\n', '    for i in range(int(threadnum)):\n', '        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))\n', '    """image list to be run """\n', '    img = []\n', '    for i in range(runTotall):\n', '        path = os.path.join(calib_image_dir, listimage[i])\n', '        img.append(preprocess_one_image_fn(path, height=height, width=width))\n', '\n', '    cnt = 1\n', '    """run with batch """\n', '    i=1\n', '    for x in range(10):\n', '        i=2\n', '        print("Start Inferenz")\n', '        time_start = time.time()\n', '        for i in range(int(threadnum)):\n', '            t1 = threading.Thread(target=runNet, args=(all_dpu_runners[i], img, cnt))\n', '            threadAll.append(t1)\n', '        for x in threadAll:\n', '            x.start()\n', '        for x in threadAll:\n', '            x.join()\n', '        time_end = time.time()\n', '        timetotal = time_end - time_start\n', '        total_frames = cnt * int(threadnum)\n', '        fps = float(total_frames/timetotal)\n', "        #timefile = open('./time.txt','a')\n", '        #timefile.write(str(timetotal)+"\\n")\n', '        #timefile.close()\n', '        time.sleep(50/1000)\n', '        threadAll = []\n', '        print("FPS=%.2f, total frames = %.2f, time=%.6f seconds" %(fps, total_frames, timetotal))\n', '    del all_dpu_runners\n', '    \n', '\n', '    time_end = time.time()\n', '    timetotal = time_end - time_start\n', '    total_frames = cnt * int(threadnum)\n', '    fps = float(total_frames / timetotal)\n', '    print(\n', '        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"\n', '        % (fps, total_frames, timetotal)\n', '    )\n', '\n', '\n', 'if __name__ == "__main__":\n', '    if len(sys.argv) != 3:\n', '        print("usage : python3 resnet50.py <thread_number> <resnet50_xmodel_file>")\n', '    else:\n', '        main(sys.argv)\n']


########Config##########

VART_WORKSPACE = './VART_WORKSPACE/'
SAVE_FILES = './savedfiles/'

server = "192.168.1.18"
username = "root"
password = "root"
xilinx_image = '/home/mwuschnig/Standard_image.json'

duration_measurement = 2
datalogger_port = 3
############################


directory = os.getcwd()
print(directory)


def check_config():
        if not(os.path.isfile(xilinx_image)):
            raise FileNotFoundError("Xilinx image missing")
        print("Config seems to be good")
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(server,username=username,password=password)
            ssh.close()
        except:
            print("SSH Config not good")
            #print(sys.exc_info()[0])
            raise
        else:
            print("SSH Config good")
        return(True)
        if(os.path.isdir(VART_WORKSPACE)):
              print("Remove "+VART_WORKSPACE)
              raise


check_config()

def create_workspace():
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)

        ssh.exec_command("mkdir workspace")
        time.sleep(1)
        ssh.exec_command("mkdir workspace/images")
        time.sleep(1)
        ssh.close()
        return

def mod_inferenz_base(w,h):
        inferenz_base[15] = "width = "+str(w)+"\n"
        inferenz_base[16] = "height = "+str(h)+"\n"
        os.mkdir(VART_WORKSPACE+"inferenz")
        inferenz_file = open(VART_WORKSPACE+"inferenz/inferenz.py", 'w')
        inferenz_file.writelines(inferenz_base)
        inferenz_file.close()
        return

def mod_inferenz_file(w,h):
        inferenz_base = open(VART_WORKSPACE+'/inferenz/inferenz_base.py' ,'r')
        inferenz_base_lines = inferenz_base.readlines()
        inferenz_base_lines[31] = "width = "+str(w)+ "\n"
        inferenz_base_lines[32] = "height ="+str(h)+ "\n"
        inferenz_file = open('./inferenz/inferenz.py', 'w')
        inferenz_file.writelines(inferenz_base_lines)
        inferenz_file.close()
        inferenz_base.close()
        return

def copy_inf_json_files_model(modelname):
        trace_json_file = open(VART_WORKSPACE+'trace.json','w')
        trace_json_file.writelines(trace_json)
        trace_json_file.close()


        print("copy files")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)

        ftp_client = ssh.open_sftp()
        ftp_client.put(VART_WORKSPACE+'vai_c_output/'+modelname+'/compiled.xmodel', './workspace/compiled_model.xmodel')
        ftp_client.put(VART_WORKSPACE+'/inferenz/inferenz.py', './workspace/inferenz.py')
        ftp_client.put(VART_WORKSPACE+'trace.json', './workspace/trace.json')
        ftp_client.put(VART_WORKSPACE+'calib/0.jpg', './workspace/images/test.jpg')
        time.sleep(5)
        ftp_client.close()
        ssh.close()
        return


def run_vaitrace(modelname):
        print("run vaitrace")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)
        channel = ssh.invoke_shell()
        stdin = channel.makefile('wb')
        stdout = channel.makefile('rb')
        pm = measurement.power_measurement(sampling_rate=500000, # set the sampling rate to whatever your device supports
                                    data_dir = SAVE_FILES+modelname, # pass the folder where data files will be saved
                                    max_duration=60, # set the maximal duration of the gathering process [seconds]
                                    port=datalogger_port) # if your device has more than one port, choose the current port


        test_kwargs = {"model_name" : modelname}

        pm.start_gather(test_kwargs) # start the data aquisition

        stdin.write('''
        cd /home/root/workspace
        python3 -m vaitrace_py -c ./trace.json
        exit
        ''')
 
        time.sleep(duration_measurement)
        pm.end_gather(True)
        output = stdout.read().decode('utf-8')
        stdin.close()
        stdout.close()
        channel.close()
        ssh.close()
        return

def run_time(modelname):
        print("run time")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)
        channel = ssh.invoke_shell()
        stdin = channel.makefile('wb')
        stdout = channel.makefile('rb')
        pm = measurement.power_measurement(sampling_rate=500000, # set the sampling rate to whatever your device supports
                                    data_dir = SAVE_FILES+modelname, # pass the folder where data files will be saved
                                    max_duration=60, # set the maximal duration of the gathering process [seconds]
                                    port=datalogger_port) # if your device has more than one port, choose the current port


        test_kwargs = {"model_name" : modelname}

        pm.start_gather(test_kwargs) # start the data aquisition

        stdin.write('''
        cd /home/root/workspace
        python3 inferenz.py 1 compiled_model.xmodel
        exit
        ''')
        time.sleep(duration_measurement)
        pm.end_gather(True)
        output = stdout.read().decode('utf-8')
        stdin.close()
        stdout.close()
        channel.close()
        ssh.close()
        return



def delete_workspace():
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)
        ssh.exec_command("rm -rf workspace/")
        time.sleep(1)
        ssh.close()
        return

def get_timefile(modelname):
        print("get_timefile")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)

        ftp_client = ssh.open_sftp()
        try:
            os.mkdir(SAVE_FILES)
        except:
            print("folder exists")
        try:
            os.mkdir(SAVE_FILES+'/'+modelname)
        except:
            print("folder exists")
        ftp_client.get('./workspace/time.txt', SAVE_FILES+'/'+modelname+'/time.txt')
        ftp_client.close()
        ssh.close()
        return



def get_traces(modelname):
        print("get_traces")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server,username=username,password=password)

        ftp_client = ssh.open_sftp()
        try:
            os.mkdir(SAVE_FILES)
        except:
            print("folder exists")
        try:
            os.mkdir(SAVE_FILES+'/'+modelname)
        except:
            print("folder exists")
        ftp_client.get('./workspace/profile_summary.csv', SAVE_FILES+'/'+modelname+'/profile_summary.csv')
        ftp_client.get('./workspace/vart_trace.csv', SAVE_FILES+'/'+modelname+'/vart_trace.csv')
        ftp_client.get('./workspace/vitis_ai_profile.csv', SAVE_FILES+'/'+modelname+'/vitis_ai_profile.csv')
        ftp_client.get('./workspace/time.txt', SAVE_FILES+'/'+modelname+'/time.txt')
        
        #ftp_client.get('./workspace/hal_host_trace.csv', VART_WORKSPACE+'+modelname+'/hal_host_trace.csv')
        #ftp_client.get('./workspace/xclbin.ex.run_summary', VART_WORKSPACE+'/'+modelname+'/xclbin.ex.run_summary')
        #ftp_client.get('./inferenz/inferenz.py', './workspace/inferenz.py')
        #ftp_client.get('./json_files/trace.json', './workspace/trace.json')

        ftp_client.close()
        ssh.close()
        return


def gen_images(h,w,number,directory):
  print("gen images")
  if not(os.path.isdir(directory)):
    os.mkdir(directory)
  images = []
  input_shape = (int(h), int(w), 3)
  for x in range(number):
    image = np.random.randint(255, size=input_shape, dtype=np.uint8)
    im = Image.fromarray(image)
    im.save(directory+'/'+str(x)+'.jpg')
  return


def remove_vart_workspace():
  print('remove temp workspace')
  cmd = 'rm -rf '+VART_WORKSPACE
  os.system(cmd)
  return


def optimize_network(**kwargs):
    #Build Workspace
    cwd = os.getcwd()
    os.mkdir(VART_WORKSPACE)
    os.system('cp '+kwargs['place_pb_file'] +' '+ VART_WORKSPACE+'model.pb')
    kwargs['place_pb_file'] = "./model.pb"
    os.system(f"cp {xilinx_image} ./VART_WORKSPACE/Standard_image.json")
    my_input_fn_file = open(VART_WORKSPACE+'/my_input_fn.py', 'w')
    my_input_fn_file.writelines(fn_file)
    my_input_fn_file.close()
    docker_methods_file = open(VART_WORKSPACE+'/methods_docker.py','w')
    docker_methods_file.writelines(docker_methods)
    docker_methods_file.close()

    #Start Docker Container
    print("start dockercommand")
    start_docker_command[22] = cwd+VART_WORKSPACE.strip('.')+':/workspace'
    #print(start_docker_command)
    docker_process = subprocess.run(start_docker_command, stdout = subprocess.PIPE)
    string = docker_process.stdout.decode("utf-8")
    docker_name = string.rstrip()
    gen_images(kwargs['width'],kwargs['height'],2,"./VART_WORKSPACE/calib")

    # Get Input Output Nodes and write into my_input_fn.py
    #if not(("input_node" in kwargs.keys())and("output_node" in kwargs.keys())):
    if((kwargs['input_node']==None)or(kwargs['output_node']==None)):
        cmd = f"docker exec {docker_name} bash -c 'source /opt/vitis_ai/conda/etc/profile.d/conda.sh;conda activate vitis-ai-tensorflow;cd /workspace;python3 methods_docker.py --pb {kwargs['place_pb_file']} --execute get_nodes '"
        os.system(cmd)
        with open(VART_WORKSPACE+'/my_input_fn.py') as file:
            all_lines = file.readlines()
            for line in all_lines:
                if("input_node =" in line):
                    kwargs['input_node'] = line.split("'")[1]
                    print(kwargs)
                if("output_node =" in line):
                    print(line)
                    kwargs['output_node'] = line.split("'")[1]
    print(kwargs)
    

    print("quantize docker")
    cmd = f"docker exec {docker_name} /bin/bash -c 'source /opt/vitis_ai/conda/etc/profile.d/conda.sh;conda activate vitis-ai-tensorflow;cd /workspace;python3 methods_docker.py --pb {kwargs['place_pb_file']} --execute quantize --name {kwargs['modelname']} --save-folder ./  --input_node {kwargs['input_node']} --output_node {kwargs['output_node']} --height {kwargs['height']} --width {kwargs['width']}'"
    os.system(cmd)


    print("compile docker")
    cmd = f"docker exec {docker_name} /bin/bash -c 'source /opt/vitis_ai/conda/etc/profile.d/conda.sh;conda activate vitis-ai-tensorflow;cd /workspace;vai_c_tensorflow -f vai_q_output/{kwargs['modelname']}/quantize_eval_model.pb -a Standard_image.json -o vai_c_output/{kwargs['modelname']} -n compiled'"
    os.system(cmd)

    cmd = f"docker exec {docker_name} /bin/bash -c 'cd /workspace; chmod -R 777 vai_q_output; chmod -R 777 vai_c_output; chmod -R 777 __pycache__'"
    os.system(cmd)

    print("kill docker:")
    cmd = f"docker kill {docker_name}"
    os.system(cmd)

    return()

def run_network(**kwargs):
  mod_inferenz_base(kwargs['width'],kwargs['height'])
  create_workspace()
  copy_inf_json_files_model(kwargs['modelname'])
  if(kwargs['execution']=='Trace'):
    run_vaitrace(kwargs['modelname'])
    get_traces(kwargs['modelname'])
  if(kwargs['execution']=='Time'):
    run_time(kwargs['modelname'])
    get_timefile(kwargs['modelname'])
  delete_workspace()
  remove_vart_workspace()




  return()




if __name__ == '__main__':
      
  check_config()

  parser = argparse.ArgumentParser(description='ZCU102 inference benchmark')
  parser.add_argument("-p", "--place_pb_file", default="./models/annete/annette_bench1.pb")
  parser.add_argument('-fw', '--framework', default='tf')
  parser.add_argument('-m','--modelname', default='network')
  parser.add_argument('-x','--height', default= 1)
  parser.add_argument('-w', '--width', default = 1)
  parser.add_argument('-c', '--channels', default = 3)
  parser.add_argument('-b', '--batch' , default = 1)
  parser.add_argument('-i', '--input_node')
  parser.add_argument('-o', '--output_node')
  parser.add_argument('-e', '--execution', default = 'Time')
  args = parser.parse_args()
  dictonary = vars(args)

  optimize_network(**dictonary)
  run_network(**dictonary)


