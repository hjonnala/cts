# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coral Compatibility Test Suite.

The Coral Compatibility Test Suite is intended for validating partner hardware
operates as expected. It runs a variety of long-running tests and benchmarks.
The output is provided on stdout as well as to a file, passed in with the
--output argument (by default cts.txt).

In order to be approved for Coral branding, all tests must pass.

python3 coral_cts.py --test_name tflite_utils_test
python3 coral_cts.py --test_name inference_stress_test
python3 coral_cts.py --test_name model_loading_stress_test
python3 coral_cts.py --test_name inference_repeatability_test
python3 coral_cts.py --test_name classification_models_test
python3 coral_cts.py --test_name detection_models_test
python3 coral_cts.py --test_name segmentation_models_test
python3 coral_cts.py --test_name multiple_tpus_inference_stress_test
python3 coral_cts.py --test_name models_benchmark
"""

import argparse
import os
import platform
import subprocess
import ssl
import sys

from shutil import unpack_archive
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

CTS_HEADER = """#####################################################
        Coral Compatibility Test Suite
#####################################################\n\n"""

SECTION_HEADER = "-----------------------------------------------------\n"


class TestSuite():
    """Helper class for running tests and storing results.

    Attributes:
        results: A dictionary of tests and their results.
        file_path: Location of file (absolute path).
        file: File handle for test output.
        tpus: Number of TPUs detected on the system.
        pci: At least one TPU is connected over PCIe.
        usb: At least one TPU is connected over USB.
        thermals: A dictionary of tests and the max recorded temperature.
    """

    def __init__(self, file_path):
        self.results = dict()
        self.file_path = file_path
        self.file = open(file_path, "w")
        self.tpus = 0
        self.pci = False
        self.usb = False
        self.thermals = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._write_summary()

    def _file_print(self, message):
        """Helper function that prints to both the console and a file.

        Args:
            message: A string containing the message.
        """
        self.file.write(message)
        self.file.flush()
        sys.stdout.write(message)

    def _run_linux_system_command(self, header, command, output_parse=""):
        """Helper function that runs a linux command.

        Args:
            header: Header string for the section
            command: Command and args to be passed to Popen
            output_parse: String used to parse output to useful data.
        """
        self._file_print("\n***** " + header + " *****\n")
        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                if not output_parse or output_parse in line:
                    self._file_print(line)
            proc.wait()
        except Exception as e:
            self._file_print(str(e) + "\n")

    def _write_thermal_summary(self):
        self._file_print("\n" + SECTION_HEADER)
        self._file_print("Temperatures During Tests" + "\n")
        self._file_print(SECTION_HEADER)
        for test in self.thermals:
            self._file_print(test + ": " + str(self.thermals[test]) + "\n")
        self._file_print("\n")

    def _write_summary(self):
        """Writes the summary.

        Generates a table of the complete results of the test, and provides
        a final overall Pass/Fail. The summary is printed on the console and
        added to the beginning of the output file.
        """
        if self.thermals:
            self._write_thermal_summary()

        # In order to prepend the summary, closes the file, creates a new
        # temporary file with the header, and copies in the old contents.
        self.file.close()
        temp_path = self.file_path + ".tmp"
        summary = CTS_HEADER
        with open(self.file_path, "r") as file:
            with open(temp_path, "w") as temp_file:
                overall_passed = True
                for test in self.results:
                    summary += (test + ": " +
                                ("Passed" if self.results[test] is True else "Failed") + "\n")
                    overall_passed = overall_passed and self.results[test]
                if self.thermals:
                    max_temp = max(self.thermals.values())
                    summary += "\nMax Temperature (C): " + str(max_temp)
                summary += "\nOverall Compatibility: " + \
                    ("Passed" if overall_passed is True else "Failed") + "\n"
                print("\n" + summary)
                temp_file.write(summary)
                temp_file.write("\n\n")
                temp_file.write(file.read())
        os.rename(temp_path, self.file_path)

    def _read_temperatures(self, current_test):
        command = ["cat /sys/class/apex/apex_*/temp"]
        temperatures = []
        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
            for line in proc.stdout:
                temperatures.append(float(line) / 1000.0)
            proc.wait()
        except Exception as e:
            pass
        if temperatures:
            self.thermals[current_test] = max(temperatures)

    def run_test(self, test):
        """Runs a given test.

        Runs a given test, providing output to the console and output file. The
        results are stored in a dictionary with the key being the test name and
        the value as a Boolean indicating if the test passed.

        Args:
            test: The name / relative path of the test.
        """
        current_test = test[0]
        self._file_print("\n" + SECTION_HEADER)
        self._file_print(current_test + "\n")
        self._file_print(SECTION_HEADER)
        test[0] = os.getcwd() + "/" + current_test
        try:
            proc = subprocess.Popen(
                test, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                self._file_print(line)
            proc.wait()
        except Exception as e:
            self._file_print(str(e) + "\n")
            self.results[current_test] = False
            return

        if self.pci and sys.platform == "linux":
            self._read_temperatures(current_test)

        if proc.returncode:
            self.results[current_test] = False
        else:
            self.results[current_test] = True

    def detect_tpus(self):
        """Detects number of TPUs.

        Runs lstpu, which outputs the paths of TPUs on the system.

        Returns:
            An integer that indicates the number of TPUs detected.
        """
        test = "lstpu"
        test_path = os.path.join(os.getcwd(), test)
        self._file_print("\n" + SECTION_HEADER)
        self._file_print("Detected TPUs (lstpu)\n")
        self._file_print(SECTION_HEADER)

        try:
            proc = subprocess.Popen(
                test_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                if not self.pci and "PCI" in line:
                    self.pci = True
                if not self.usb and "USB" in line:
                    self.usb = True
                self.tpus += 1
                self._file_print(line)
            proc.wait()
        except Exception as e:
            self._file_print(str(e) + "\n")

        if self.tpus:
            self.results[test] = True
        else:
            self.results[test] = False
            self._file_print("No TPUs detected\n")

        return self.tpus

    def print_system_info(self):
        """Prints system info.

        Runs various commands (currently Linux only) to print information about
        the system, including PCIe and USB when relevant.
        """
        self._file_print(
            "\n-----------------------------------------------------\n")
        self._file_print("System Info\n")
        self._file_print(
            "-----------------------------------------------------\n")
        self._file_print(platform.platform() + "\n")
        if(sys.platform == "linux"):
            if self.pci:  # For PCIe, displays apex kernel messages and lspci output.
                self._run_linux_system_command(
                    "TPU Kernel Messages", ["dmesg"], "apex")
                self._run_linux_system_command(
                    "PCI Info", ["lspci", "-vvv", "-d 1ac1:089a"])
            elif self.usb:   # For USB, the device can be in DFU mode or TPU mode.
                self._run_linux_system_command(
                    "USB Devices", ["lsusb"])
                self._run_linux_system_command("USB Tree", ["lsusb", "-t"])
                self._run_linux_system_command(
                    "USB Detailed Info (TPU in standard mode)", ["lsusb", "-v", "-d 18d1:9302"])
                self._run_linux_system_command(
                    "USB Detailed Info (TPU in DFU mode)", ["lsusb", "-v", "-d 1a6e:089a"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default="tflite_utils_test")
    parser.add_argument('--output', default="cts.txt")
    args = parser.parse_args()

    # Gets the complete path of file.
    output_file = os.path.join(os.getcwd(), f'{args.test_name}.txt')

    # Checks for and downloads/extracts test data.
    TEST_DATA_COMMIT = "c21de4450f88a20ac5968628d375787745932a5a"
    if not os.path.isdir(os.path.join(os.getcwd(), "test_data")):
        print("Test data not found, downloading...")
        context = ssl._create_unverified_context()
        with urlopen("https://github.com/google-coral/test_data/archive/" + TEST_DATA_COMMIT + ".zip", context=context) as zipresp, NamedTemporaryFile() as tfile:
            tfile.write(zipresp.read())
            tfile.seek(0)
            print("Download complete, extracting...")
            unpack_archive(tfile.name, os.getcwd(), format='zip')
        os.rename(os.path.join(os.getcwd(), "test_data-" + TEST_DATA_COMMIT),
                  os.path.join(os.getcwd(), "test_data"))

    with TestSuite(output_file) as cts:
        # Verifies TPU(s) are attached
        # tpus = cts.detect_tpus()
        # if not tpus:
        #     return
        tpus = 1

        cts.print_system_info()

        # Iterates through tests, outputting results to file and storing results.
        if args.test_name == 'tflite_utils_test':
            cts.run_test(test=["tflite_utils_test"])
        if args.test_name == 'inference_stress_test':
            cts.run_test(test=["inference_stress_test",
                               "--stress_test_runs=10000", "--stress_with_sleep_test_runs=200"])
        if args.test_name == 'model_loading_stress_test':
            cts.run_test(test=["model_loading_stress_test",
                               "--stress_test_runs=50"])
        if args.test_name == 'inference_repeatability_test':
            cts.run_test(test=["inference_repeatability_test",
                               "--stress_test_runs=1000", "--gtest_repeat=20"])
        # For classification test, omit TF2 ResNet50 - which fails on some platforms.
        if args.test_name == 'classification_models_test':
            cts.run_test(test=["classification_models_test", "--gtest_repeat=10",
                               "--gtest_filter=-*tfhub_tf2_resnet_50_imagenet_ptq*"])
        if args.test_name == 'detection_models_test':
            cts.run_test(test=["detection_models_test", "--gtest_repeat=100"])
        if args.test_name == 'segmentation_models_test':
            cts.run_test(test=["segmentation_models_test", "--gtest_repeat=100"])

        # If more than 1 TPU is attached, runs multi-TPU tests.
        if tpus > 1 and args.test_name == 'multiple_tpus_inference_stress_test':
            cts.run_test(
                test=["multiple_tpus_inference_stress_test", "--num_inferences=5000"])

        # Runs Benchmarks, which just reports results but don't compare.
        # Also note CPU scaling is not disabled (as it would require root).
        if args.test_name == 'models_benchmark':
            cts.run_test(test=["models_benchmark", "--benchmark_color=false"])


if __name__ == "__main__":
    main()
