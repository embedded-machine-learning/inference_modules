import argparse
import os, sys, threading
from os import system
from sys import stdout
from time import sleep, time
import numpy as np

uldaq_import = False
bench_over = False
try:
    from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag, ScanStatus,
                       ScanOption, create_float_buffer, InterfaceType, AiInputMode)
    print("Import of uldaq library for daq-card successful")
    uldaq_import = True
except:
    print("Could not load uldaq library for daq-card")


def clear_eol():
    """Clear all characters to the end of the line."""
    stdout.write('\x1b[2K')


def daq_setup(rate=500000, samples_per_channel=500000*10, resistor=0.1):
    """Analog input scan example."""
    global bench_over
    bench_over = False # beginning of a new cycle
    daq_device = None
    status = ScanStatus.IDLE
    # samples_per_channel (int): the number of A/D samples to collect from each channel in the scan.
    # rate (float): A/D sample rate in samples per channel per second.

    range_index = 0
    interface_type = InterfaceType.ANY
    low_channel = 0
    high_channel = 0
    scan_options = ScanOption.CONTINUOUS
    flags = AInScanFlag.DEFAULT

    # Get descriptors for all of the available DAQ devices.
    devices = get_daq_device_inventory(interface_type)
    number_of_devices = len(devices)
    if number_of_devices == 0:
        raise RuntimeError('Error: No DAQ devices found')

    print('Found', number_of_devices, 'DAQ device(s):')
    for i in range(number_of_devices):
        print('  [', i, '] ', devices[i].product_name, ' (',
              devices[i].unique_id, ')', sep='')

    descriptor_index = 0
    if descriptor_index not in range(number_of_devices):
        raise RuntimeError('Error: Invalid descriptor index')

    # Create the DAQ device from the descriptor at the specified index.
    daq_device = DaqDevice(devices[descriptor_index])

    # Get the AiDevice object and verify that it is valid.
    ai_device = daq_device.get_ai_device()

    if ai_device is None:
        raise RuntimeError('Error: The DAQ device does not support analog '
                           'input')

    # Verify the specified device supports hardware pacing for analog input.
    ai_info = ai_device.get_info()

    if not ai_info.has_pacer():
        raise RuntimeError('\nError: The specified DAQ device does not '
                           'support hardware paced analog input')

    # Establish a connection to the DAQ device.
    descriptor = daq_device.get_descriptor()
    print('\nConnecting to', descriptor.dev_string, '- please wait...')
    # For Ethernet devices using a connection_code other than the default
    # value of zero, change the line below to enter the desired code.
    daq_device.connect(connection_code=0)

    # The default input mode is DIFFERENTIAL.
    input_mode = AiInputMode.DIFFERENTIAL

    # Get the number of channels and validate the high channel number.
    number_of_channels = ai_info.get_num_chans_by_mode(input_mode)

    if high_channel >= number_of_channels:
        high_channel = number_of_channels - 1
    channel_count = high_channel - low_channel + 1

    # Get a list of supported ranges and validate the range index.
    ranges = ai_info.get_ranges(input_mode)
    if range_index >= len(ranges):
        range_index = len(ranges) - 1
    meas_range = ranges[1]

    data_dir = "data_dir"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # define name of a datafile and delete if it exists in the data directory
    data_fname = "test_data"
    #if os.path.isfile(os.path.join(data_dir, data_fname)):
        #os.remove(os.path.join(data_dir, data_fname))

    # Allocate a buffer to receive the data.
    data = create_float_buffer(channel_count, samples_per_channel)
    # ranges[1] = Range.BIP5VOLTS

    return daq_device, low_channel, high_channel, input_mode, meas_range, samples_per_channel,  rate, scan_options, flags, data, data_dir, data_fname


def daq_measurement(daq_device, low_channel, high_channel, input_mode,
                    meas_range, samples_per_channel,
                    rate, scan_options, flags, data, data_dir, data_fname, index_run, api, niter, nireq):
    # Start the acquisition.
    global bench_over
    ai_device = daq_device.get_ai_device()

    rate = ai_device.a_in_scan(low_channel, high_channel, input_mode,
                               meas_range, samples_per_channel,
                               rate, scan_options, flags, data)
    index = 0
    while not bench_over:
        # Get the status of the background operation
        status, transfer_status = ai_device.get_scan_status()
        # get current index
        index = transfer_status.current_index
        # when the index has reached maximal length

        #print("{} {}".format(samples_per_channel, index), end="\r", flush=True)

    # save data
    #print("jumped to break")
    with open(os.path.join(data_dir, "_".join((data_fname, str(index_run), api, "n" + str(niter), "ni" + str(nireq) +".dat"))), "wb") as f:
        np.save(f, np.asarray(data[0:index]))

    # Stop the acquisition if it is still running, disconnect
    if daq_device:
        if status == ScanStatus.RUNNING:
            ai_device.scan_stop()
        if daq_device.is_connected():
            daq_device.disconnect()
        daq_device.release()

