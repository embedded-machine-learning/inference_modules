# this file will contain a class, where all the attributes needed to setup the daq card and run measurements will be placed
import threading, os
import numpy as np

try:
    from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag, ScanStatus,
                       ScanOption, create_float_buffer, InterfaceType, AiInputMode)
    print("Import of uldaq library for daq-card successful")
    uldaq_import = True
except:
    print("Could not load uldaq library for daq-card")
    uldaq_import = False


class power_measurement():
    def __init__(self, sampling_rate=500000, data_dir = "data_dir", max_duration = 60):
        """Function for initializing default values for the DAQ-Device

        Args:
            sampling_rate: sampling rate in samples per second, maximum is 500000
            data_dir: name of the directory where the data files will be stored
            max_duration: maximal duration of the measurement in seconds
        """
        self.dev_init = False
        if uldaq_import:

            # define default values
            self.len_dev = 0
            self.bench_end = False
            self.bench_over = False
            self.daq_device = None
            self.status = ScanStatus.IDLE
            self.data_dir = data_dir

            self.range_index = 0
            self.low_channel = 0
            self.high_channel = 0
            self.scan_options = ScanOption.CONTINUOUS
            self.flags = AInScanFlag.DEFAULT

            # The default input mode is DIFFERENTIAL (can be SINGLE_ENDED)
            self.input_mode = AiInputMode.DIFFERENTIAL

            self.dat_filename = ""
            self.dat_filepath = "test"
            self.model_name = ""
            self.index_run = 0
            self.api = "sync"
            self.niter = 1
            self.nireq = 1
            self.batch = 1

            if sampling_rate > 500000:
                print("sampling rate cannot be larger 500 kS/s! Defaulting to 500 kS/s.")
                self.sampling_rate = 500000
            elif sampling_rate < 0:
                print("sampling rate cannot be negative! Defaulting to 1 S/s.")
                self.sampling_rate = 1
            else:
                self.sampling_rate = sampling_rate

            self.total_samples = self.sampling_rate * max_duration # total samples defines the length of data buffer
            self.data = None # array to contain measurement data

            print("Initialized Power Measurement Class.")

            self.setup()

    def setup(self):
        """Sets all necessary variables for DAQ-Device operation

        Returns: None

        """
        interface_type = InterfaceType.ANY
        # Get descriptors for all of the available DAQ devices.
        devices = get_daq_device_inventory(interface_type)
        self.len_dev = len(devices)
        if self.len_dev == 0:
            print('Error: No DAQ devices found')
            return False

        print('Found', self.len_dev, 'DAQ device(s):')
        for i in range(self.len_dev):
            print('  [', i, '] ', devices[i].product_name, ' (',
                  devices[i].unique_id, ')', sep='')

        descriptor_index = 0
        if descriptor_index not in range(self.len_dev):
            raise RuntimeError('Error: Invalid descriptor index')

        # Create the DAQ device from the descriptor at the specified index.
        self.daq_device = DaqDevice(devices[descriptor_index])

        # Get the AiDevice object and verify that it is valid.
        ai_device = self.daq_device.get_ai_device()

        if ai_device is None:
            raise RuntimeError('Error: The DAQ device does not support analog '
                               'input')

        # Verify the specified device supports hardware pacing for analog input.
        ai_info = ai_device.get_info()

        if not ai_info.has_pacer():
            raise RuntimeError('\nError: The specified DAQ device does not '
                               'support hardware paced analog input')

        # Establish a connection to the DAQ device.
        descriptor = self.daq_device.get_descriptor()
        print('\nConnecting to', descriptor.dev_string, '- please wait...')
        # For Ethernet devices using a connection_code other than the default
        # value of zero, change the line below to enter the desired code.
        self.daq_device.connect(connection_code=0)
        print('\nConnection with', descriptor.dev_string, '- established!')

        # Get the number of channels and validate the high channel number.
        number_of_channels = ai_info.get_num_chans_by_mode(self.input_mode)

        if self.high_channel >= number_of_channels:
            self.high_channel = number_of_channels - 1
        channel_count = self.high_channel - self.low_channel + 1

        # Get a list of supported ranges and validate the range index.
        ranges = ai_info.get_ranges(self.input_mode)
        if self.range_index >= len(ranges):
            range_index = len(ranges) - 1
        self.meas_range = ranges[1] # ranges[1] = Range.BIP5VOLTS

        # Allocate a buffer to receive the data.
        self.data = create_float_buffer(channel_count, self.total_samples)

        # make sure the data directory exists
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.dev_init = True
        return self.dev_init

    def gather_data(self, kwargs_pm):
        """Start the data acquisition. Fill the data buffer until bench_end is set to True. Save data to file.

        Returns: None

        """

        print("Starting the power measurement")
        ai_device = self.daq_device.get_ai_device()

        rate = ai_device.a_in_scan(self.low_channel, self.high_channel, self.input_mode,
                                   self.meas_range, self.total_samples,
                                   self.sampling_rate, self.scan_options, self.flags, self.data)
        index = 0
        while not self.bench_end:
            # Get the status of the background operation
            status, transfer_status = ai_device.get_scan_status()
            # get current index
            index = transfer_status.current_index

        # save data
        self.dat_filename = "_".join([str(v) for k,v in kwargs_pm.items()]) + ".dat"
        self.dat_filepath = os.path.join(self.data_dir, self.dat_filename)
        print("Writing power measurement data to %s" % str(self.dat_filepath))

        with open(self.dat_filepath, "wb") as f:
            print("Saving", index, "data points.")
            np.save(f, np.asarray(self.data[0:index])) # save only valid data (up until index)

        # Stop the acquisition if it is still running, disconnect
        print("Releasing the device handle")
        if self.daq_device:
            if status == ScanStatus.RUNNING:
                ai_device.scan_stop()
            if self.daq_device.is_connected():
                self.daq_device.disconnect()
            self.daq_device.release()
        print("Power measurement ended")
        self.bench_over = True

    def end_bench(self, b_val):
        # calling this function with a True ends the data acquisition
        if self.dev_init:
            self.bench_end = b_val
            while not self.bench_over:
                pass

    def get_data_fname(self):
        return self.dat_filename

    def get_data_fpath(self):
        return self.dat_filepath

    def start_gather(self, kwargs_pm):
        if self.dev_init:
            t_pm = threading.Thread(target=self.gather_data, kwargs={"kwargs_pm":kwargs_pm})
            t_pm.start()


if __name__ == "__main__":
    pm = power_measurement(sampling_rate=500000, data_dir = "data_dir", max_duration=60)

    #print(pm.__dict__)
    test_kwargs = {"model_name" : "awesome_model", "index_run" : 69, "api" : "async", "niter" : 420, "nireq" : 2, "batch" : 2}

    pm.start_gather(test_kwargs)

    from time import sleep
    sleep(2)
    pm.end_bench(True)
    print("Finished")