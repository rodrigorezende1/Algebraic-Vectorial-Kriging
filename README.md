# Algebraic-Vectorial-Kriging 💡

This repository provides a Julia implementation of Algebraic Vectorial Kriging (AVK), a novel vectorial surrogate model designed to accelerate optimization procedures that rely on expensive simulations.

The primary goal of this project is to address the prohibitive computational cost of these simulations. We implement a novel vector-valued Gaussian Process (GP), specifically designed to handle the vectorial outputs of problems (e.g., in computational electromagnetics or fluid dynamics). By relying on a small number of high-fidelity data points, this GP surrogate can predict simulation results with high accuracy, achieving speedups of several orders of magnitude in benchmark optimization problems.

---------------------------------

== ✨ Features ==

* Novel Surrogate Modeling: Implements Algebraic Vectorial Kriging and Co-Kriging, novel vector-valued Gaussian Process (GP) methods.
* Standard Methods: Includes standard 1D Kriging and Co-Kriging models for comparison and benchmarking.
* Performance Evaluation: Provides functions to compute the accuracy and performance of the surrogate models.
* Experiment Tools: Contains built-in test functions and utilities for generating sampling plans.
* Built in Julia: Leverages the Julia programming language for high performance and scientific computing.

---------------------------------

== 🧠 Core Concepts ==

The Challenge: Expensive Simulations
In many engineering fields, optimization procedures (e.g., shape optimization) rely on repeated, high-fidelity simulations. The computational cost of running thousands of these simulations is often prohibitively expensive, making thorough optimization impractical.

The Solution: Algebraic Vectorial Kriging (AVK)
This project implements AVK, a cutting-edge surrogate model (or "metamodel") that learns from a small number of expensive simulation runs.
* It's a Gaussian Process: It models a distribution over functions, allowing it to provide not only a prediction but also a confidence interval (variance).
* It's Vector-Valued: Unlike standard GPs that predict a single scalar, AVK is specifically designed to handle the complex, high-dimensional vectorial outputs common in mesh-based simulations.

By replacing the "real" simulation with this fast-to-evaluate surrogate, the AVK model can enable optimization speedups of several orders of magnitude.

---------------------------------

== 📂 Project Structure ==

.
├── Co_Kriging/src/         # Implementation of Co-Kriging models (incl. AV-CoKriging)
├── ErrorCalc/src/          # Functions to compute model accuracy
├── Kriging/src/            # Implementation of Kriging models (incl. AVK)
├── Sampling_Plan/src/      # Utilities for generating sampling plans
├── Testing_Functions/src/  # Benchmark test functions
├── LICENSE                 # Project License
├── README.md               # This README file

---------------------------------

== 🚀 Getting Started ==

Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Ensure you have Julia installed (e.g., version 1.6 or higher). You can download it from julialang.org (https://julialang.org/).

2. Clone the Repository

   git clone https://github.com/rodrigorezende1/Algebraic-Vectorial-Kriging.git
   cd Algebraic-Vectorial-Kriging

3. Install Dependencies (TO DO*)

   1. Launch the Julia REPL by typing 'julia' in your terminal.
   2. Enter the package manager by pressing ']'.
   3. Activate the project environment and install the required packages:

      (v1.x) pkg> activate .
      (Algebraic-Vectorial-Kriging) pkg> instantiate

This will download and install all required packages. Press <backspace> to return to the Julia prompt.

4. Run an Example (TO DO*)

To run one of the built-in test functions, you can 'include' the main script from within the Julia REPL. For example:

   julia> include("Testing_Functions/src/run_benchmark.jl")
