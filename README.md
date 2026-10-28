# Efficient Moment Propagation for BNNs

This is the code for the paper **"Sustainable Uncertainty Aware AI through Moment Propagation in Bayesian Neural Networks"**.

This project uses the `BayesianTorch` library to run Bayesian Neural Networks (BNNs) on GPUs using the Moment Propagation (MP) algorithm. It also includes a tool that generates C++ code so models can be run efficiently on CPUs.

## Project Structure

- `requirements.txt`: Required Python packages.
- `empbnn/`: The main Python package for running models on the GPU and generating C++ code.
  - `model.py`: Base classes for the models.
  - `layers.py`: Base classes for the network layers.
  - `torch.py`: PyTorch versions of the MP layers and tools to convert `BayesianTorch` models.
  - `codegen.py`: Creates the C++ code from the Python models.
  - `metrics.py`: Calculates accuracy, uncertainty, and analyzes predictions.
  - `sources_cpp/`: Base C++ code used to build the CPU code.
    - `tensorlib/`: A C++ library for handling math and tensors.
- `models.py`: The test models used in the paper.
- `train.py`: Script to train and test the models on the GPU.
- `c_run.py`: Generates C++ code from trained models and runs them on the CPU.
- `profiling.py`: Measures how fast the models run on the GPU and CPU.
- `profiling.sh`: Sets up the GPU to ensure stable tests, then runs `profiling.py`.
- `select_strategy.py`: Selects the most energy efficient inference strategy that meets accuracy and uncertainty limits.
- `Model/`: Directory where trained models are saved.
- `Data/`: Directory for the datasets.
- `Code/`: Directory where the generated C++ code is saved.
- `Predictions/`: Directory where the test results and predictions are saved.
