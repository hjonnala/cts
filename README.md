# Coral Compatibility Test Suite

The Coral Compatibility Test Suite (CTS) allows Coral partners to evaluate
their hardware designs that contain Coral devices. The test acts as both
a validation test and a stress test. Passing CTS indicates the device works
as expected and performance/reliability matches official Coral platforms.

CTS is required for launching products branded "Made with Coral Intelligence".
For more information on marketing and branding, see
[Approval Requirements](#approval-requirements).

## What's in the CTS
The CTS is a collection of pre-compiled tests, all taken from
[libcoral](https://github.com/google-coral/libcoral). The tests have been
compiled for multiple CPU architectures (ARM64, ARMv7a, and x86_64) as well
as multiple OSes (Linux, Windows, MacOS).

The test includes:

* Inference stress testing
* Model loading stress testing
* Accuracy testing for classification, detection, and segmentation.
* Multi-TPU / Pipelining tests for devices with multiple TPUs.
* Benchmarking

## Getting CTS Packages
It is not necessary to clone this entire repo, rather just download the
appropriate archive for each architecture and OS you need to test. More
info about which packages are needed for approval can be found in
[Approval Requirements](#approval-requirements).

The release packages can be found in the [GitHub Releases](https://github.com/google-coral/cts/releases/).

If your system doesn't have a package, you can build the appropriate
libcoral binaries.
See [Building for Other Platforms](#building-for-other-platforms).

## Running CTS

The first step is to download the appropriate archive for your test. Each
archive contains the compiled binaries and the coral_cts.py script.

### Preparing for the test
The test is designed to have as few system requirements as possible. The
binaries are all statically built with the Coral libraries, so all that is
needed is libusb (for USB) or the PCIe driver (see
[installation steps](https://coral.ai/docs/m2/get-started/#2-install-the-pcie-driver-and-edge-tpu-runtime)).

It is recommended that before running the test, you set your CPU governor to
max performance to ensure consistent benchmark readings (and for better
comparison to the Coral boards, included in the
[example_outputs](example_outputs) folder).
In Linux this can be done by running:
```
sudo apt install linux-cpupower
sudo cpupower frequency-set --governor performance
```

Once you have the archive, you're ready to extract it and run. Note that the
test begins with downloading
[Coral Test Data](https://github.com/google-coral/test_data/tree/c21de4450f88a20ac5968628d375787745932a5a) (checked out at at the Frogfish Release) 
if there is not a folder named "test_data" in the extracted directory. Test data
is the entire repository of Coral models, it takes up > 1 GB of storage. If
you already have this repo on your system and want to skip the download, simply
link the folder into the working directory with coral_cts.py.

### Running the test
The test runs in python, with no extra packages needed. To run (note: this assumes that USB/PCIe access 
is available to the current user, if not please use sudo or run as an administrator):
```
python3 coral_cts.py
```

There is only one argument, `--output`, which sets the output txt file. By
default, this is `cts.txt` in the current directory.

## Approval Requirements
In order to be approved for "Made with Coral Intelligence" branding, you should
first talk with your Coral contact or reach out on the
[Coral Sales Form](https://g.co/coral/sales) for parternship, co-marketing, and legal questions.

Once it's time to run CTS, the requirements will vary based on the platform
you have designed (and what it's intended to pair with):

**PCIe Card (M.2, mPCIe, x1):**

You will need to run three different CTS configurations: ARM64 Linux,
x86_64 Linux, and x86_64 Windows.

**PCIe Card (x16)**

You will need to run two different CTS configurations: x86_64 Linux
and x86_64 Windows.

**USB Accessory**

You will need to run three different CTS configurations: ARM64 Linux,
x86_64 Linux, and x86_64 Windows. x86_64 macOS is also recommended,
but is not required.

**End product**

If the design is a end product (i.e. not a an accessory intended to pair with
other hosts), CTS is just run for the OS/Arch of the system (for example, an
IP camera with Coral inside that runs on 64-bit ARM Linux would only need to
run that test).

## Building for Other Platforms
It is possible that your system doesn't work with the included binaries. You
will need to build the `examples`, `benchmarks`, and `tests` targets
from the [libcoral](https://github.com/google-coral/libcoral) repo.

Once built, your outputs will be in the `out` folder of libcoral. You'll want
to get the following binaries:
```
From out/<arch>/examples/coral:
lstpu

From out/<arch>/tests/coral:
tflite_utils_test
inference_stress_test
model_loading_stress_test
inference_repeatability_test
segmentation_models_test
multiple_tpus_inference_stress_test

From out/<arch>/tests/coral/classification:
classification_models_test

From out/<arch>/tests/coral/detection:
models_test (this must be renamed to detection_models_test)

From out/<arch>/benchmarks/coral/:
models_benchmark
```

Place these binaries in the same folder as coral_cts.py and run the script as
described above.

## Issues
If you have technical issues, please reach out to coral-support@google.com.

For marketing/partnership questions, please reach out to your Coral contact or
the [Coral Sales Form](https://g.co/coral/sales).


