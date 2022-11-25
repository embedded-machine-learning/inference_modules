# this script calls the inference modules on a remote RPi4 ubuntu aarch64 server

from powerutils import measurement
import time, os, paramiko

def test_rpi4_ssh():
    # call this function from nuc
    # public ssh key of nuc is in rpi4 authorized_keys
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=4, range_index=-1)

    # execute script on remote machine
    ssh_ip = "192.168.1.13"
    ssh_username = "ubuntu"
    ssh_command_execute_nets = "source /home/ubuntu/imatvey/inference_modules/venv_infmd/bin/activate &&" \
                               " python3 /home/ubuntu/imatvey/inference_modules/tests/test_rpi4.py"

    # initialize and connect ssh
    client = paramiko.SSHClient()
    print("setup client")
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
    print("set host policy")
    client.connect(ssh_ip, port=22, username=ssh_username)
    print("connected")
    stdin, stdout, stderr = client.exec_command("ls /home/ubuntu/imatvey/")  # execute remote command
    # print command results
    for line in stdout:
        try:
            print('... ' + line.strip('\n'))
        except KeyboardInterrupt:
            print("exited on line:", line)

    test_kwargs = {"hardware": "rpi4_ubuntu_server_aarch64_run_tests"}
    pm.start_gather(test_kwargs)
    stdin, stdout, stderr = client.exec_command(ssh_command_execute_nets)  # execute remote command

    # print command results
    for line in stdout:
        try:
            print('... ' + line.strip('\n'))
        except KeyboardInterrupt:
            print("exited on line:", line)

    client.close()  # close the ssh channel

    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"

def main():
    test_rpi4_ssh()

if __name__ == "__main__":
    # run main
    main()