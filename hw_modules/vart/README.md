# VART Inference Module

## Requirements

Docker has to be installed 
xilinx/vitis-ai-cpu docker has to be installed Docker Image Version: 1.3.598 tested

- Python modules:
- paramiko
- numpy
- PIL
- argparse
- powerutils (ANNETTE)


## Config

start_docker_command in line 17 has to be modified to your needs/hardware

Between line 22 & 34 you have to change the configuration of the module.

VART_WORKSPACE = Temporary directory necassery workspace for the module		e.g. './VART_WORKSPACE/'

SAVE_FILES = Directory where the output files are saved				e.g. './savedfiles/'

server = IP Adress of the Xilinx Board						e.g. 192.168.1.18

username = USERNAME of the Xilinx Board						e.g. root

password = USERNAME of the Xilinx Board						e.g. root

xilinx_image = Absolute path where to find the .json file which DPU is used for building the Xilinx Image

duration_measurement = Time how long the measurement will take place		e.g. 2

datalogger_port = Port of the connected Datalogger (look at powerutils)		e.g. 3

# Functions

## check_config():
checks if the xilinx_image is found.
checks the ssh connection.
checks if the temporary VART_WORKSPACE is not already used.


## create_workspace():
creates the temporary workspace on the Xilinx Board

## mod_inferenz_base(h,w):
inserts height [h] and width [w] in the inferenz base file

## copy_inf_json_files_model(modelname):
copys the inferenz, json, model, created image to the Xilinx Board.

## run_vaitrace(modelname)
Starts the powermeasurement and start vaitrace on the Xilinx Board also ends the powermeasurement.

## run_time(modelname):
Starts the powermeasurment and starts the model on the Xilinx Board, also ends the powermeasurment.

## delete_workspace():
Delets the temporary workspace on the Xilinx Board.

## get_timefile(modelname):
Gets the time.txt from the Xilinx board and saves it to SAVE_FILES/modelname/time.txt

## get_traces(modelname):
Gets the output from vaitrace from the Xilinx board and saves it to SAVE_FILES/modelname/*vaitrace_files

## gen_images(h,w,number,directory):
create number of random images with height [h] width [w] in directory

## remove_vart_workspace()
removes the temporary created VART_WORKSPACE

## optimize_network(**kwargs)
Needs a Tensorflow 1 model (1.15)

Makes a quantised model from the given kwargs and also compiles the model.

Quantised model will be saved in VART_WORKSPACE/vai_q_output/modelname/quantize_eval_model.pb

Compiled model will be saved in VART_WORKSPACE/vai_c_output/modelname/compiled.xmodel

Needs folowing entries:
- modelname = Name of the Model
- place_pb_file = Absolute path of pb file
- width = width of input image
- height = height of input image

optional:
- input_node = Name of input node
- output_node = Name of output node

## run_network(**kwargs):
Runs the compiled model on the specified hardware (in Config).

It takes the compiled model from VART_WORKSPACE/vai_c_output/modelname

Removes the VART_WORKSPACE afterwards.

Needs following entries:
- modelname = Name of the Model
- width = Width of input image
- height = Height of input image
- execution = 'Trace' or 'Time' different kind of model execution
 


# Example for usage

Arguments:
- -p --place_pb_file = Absolute_Path_to_model
- -m --modelname = Modelname
- -x --height = height of input image
- -w --width = width of input image
- -c --chanels = number of input channels
- -b --batch = batch size of input model
- -i --input_node = input node of model (optional can be found out by optimize)
- -o --output_node = output node of model (optional can be found out by optimize)
- -e --exection = 'Trace' or 'Time' mode of exection 

e.g.

python3 inference.py -p ABSOLUTE_PATH_TO_MODEL/TF1_MODEL.pb -m testmodel -c 3 -b 1 -w 224 -x 224 -e Trace