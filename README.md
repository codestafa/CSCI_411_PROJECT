# Fourier Algorithms Implementation

This project implements three types of Fourier algorithms:

- **Discrete Fourier Transform (DFT)**
- **Fast Fourier Transform (FFT) - Recursive**
- **Fast Fourier Transform (FFT) - Vector Multiplication**

This project demonstrates the differences between DFT and two FFT implementations. By analyzing their performance, we gain a better understanding of why FFT is the preferred algorithm in most real-world applications and how implementation details can affect efficiency.

## Algorithms Implemented

### Discrete Fourier Transform (DFT)

**NOTE:** This algorithm will not be tested because the computation time is too slow.

The DFT calculates the frequency domain representation of a signal directly, with a time complexity of \(O(n^2)\). While it’s simple to implement, it’s inefficient for large inputs.

### Fast Fourier Transform (FFT) - Recursive

**NOTE:** This algorithm will not be tested because the computation time is too slow.

This is a divide-and-conquer approach to computing the Fourier transform. It reduces complexity to \(O(n \log n)\) but involves recursive function calls, which can lead to performance bottlenecks due to overhead.

### Fast Fourier Transform (FFT) - Vector Multiplication

This implementation leverages vectorized operations and modern hardware optimizations like SIMD instructions. It also processes data in contiguous blocks, improving cache performance and reducing computational overhead compared to recursion.

The inputs passed into this algorithm will be compared to output computed by the numpy FFT implementation.

## Why Compare Recursive and Vector Multiplication?

Both recursive and vector multiplication implementations of FFT achieve \(O(n \log n)\) runtime. However, the vector multiplication method is faster in practice due to:

- **Memory Efficiency:** Better utilization of contiguous memory blocks.
- **Hardware Optimizations:** Utilizes SIMD instructions and has fewer cache misses.
- **Reduced Overhead:** Avoids the overhead of recursive function calls.

These differences highlight the practical benefits of vectorized FFT for real-world applications.

Plus, I wanted to test my implementation on wave files that are over a minute long. With the recursive method, this would take way too long.

## Installation / Getting Started

To get started, follow the steps below to set up your environment and start working with the project.

### 1. Install Python

Make sure you have Python 3 installed on your machine. On Linux, you can install Python using `apt` (for Ubuntu/Debian-based systems) or another package manager for your distro.

For Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 2. Clone the Repository

Clone the project repository to your local machine.

`git clone <repository_url>`

### 3. Folder Structure

Your folder structure should look like this:

```bash
➜  project tree
.
├── README.md
├── project.py
├── sample_rates
│   └── 48000.wav
├── test_input
│   ├── 440.wav
│   ├── 440_880.wav
│   ├── impulse.wav
│   ├── mute.wav
│   ├── song.wav
│   └── whitenoise.wav
└── test_output
    ├── 440_880_output.out
    ├── 440_output.out
    ├── impulse_output.out
    ├── mute_output.out
    ├── song_output.out
    └── whitenoise_output.out

```

The `sample_rates` folder will be used to show plots when using different sample rates.
The `test_input` folder contains input files, which will be compared to the outputs in the test_output folder. The output files are the Fourier transforms computed using the numpy implementation. The inputs are all wave files which will be broken down into signals.

### 4. Create a Virtual Environment

To manage dependencies, create a virtual environment.

 - Install virtualenv if you don't have it:

 `pip3 install virtualenv`

 - Create a directory for the virtual environment:

 `mkdir venv`

 - Create the virtual environment:

 `virtualenv venv`

 ### 5. Activate the Virtual Environment

 - To activate the virtual environment, run:

 `source venv/bin/activate`

 - If you get a "Permission Denied" error, you may need to change the permissions of the activation script:

 `chmod +x venv/bin/activate`

### 6. Install Dependencies

With the virtual environment activated, install the required dependencies for the project:

`pip install -r requirements.txt`

Make sure to check the requirements.txt for the necessary libraries (e.g., numpy, scipy, etc.).

## Tests

The test input folder is `test_input`.

```
    440.wav is a signal with a single frequency.
    440880.wav is a signal with two frequencies.
    whitenoise.wav is a white noise signal.
    mute.wav is a signal of all zeros (it has no sound).
    impulse.wav tests a signal containing one non-zero value.
    song.wav is a real song that is over a minute long.

```

## Running the Program

To run the program, use the following commands:

- Run tests without generating plots (useful for only testing):

`python3 project.py test_input test_output --dont_plot`

- Run tests with generating plots for `test_input` folder:

`python3 project.py test_input test_output`

- Run tests with sample rate plot (for downsampling):

`python3 project.py test_input test_output --sample_rate_plot`


The `--dont_plot` flag prevents the program from generating plots. This is useful when you just want to check if the tests pass or fail.
The `--sample_rate_plot` flag generates plots for the 48000 Hz WAV file in the sample_rates folder to show how frequencies change as the WAV file is downsampled.


