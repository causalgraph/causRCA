
[![IWU][iwu-shield]](https://www.iwu.fraunhofer.de/)
[![doi-paper-shield]](https://github.com/causalgraph/causRCA)
[![License][apache2.0-licence]](https://opensource.org/license/apache-2-0)
[![GitHub][github-shield]](https://github.com/causalgraph/causRCA)

# causRCA - Real-World Dataset for Causal Discovery and Root Cause Analysis in Machinery

A comprehensive repository for the causRCA dataset and benchmarking framework for causal discovery and automated root cause analysis (RCA) methods in industrial machinery.

## Overview

This repository contains the implementation, evaluation scripts, and benchmarking tools for the **causRCA dataset** - a collection of real-world time series data from an industrial vertical lathe. The dataset includes both normal operation data and labeled fault scenarios from hardware-in-the-loop simulations, providing ground truth for causal discovery and root cause analysis research.

### Dataset Highlights

- **Real-world industrial data** from CNC-controlled vertical lathe
- **Labeled fault scenarios** with known ground truth causes
- **Expert-derived causal graph** for benchmarking causal discovery methods
- **Multiple fault types**: Coolant, hydraulics, and probe system failures
- **Extensive metadata** enabling comprehensive method evaluation

## Repository Structure

```
├── data/                     # causRCA dataset (see data/README_DATA_DESCRIPTION.md)
│   ├── real_op/             # Normal operation time series
│   ├── dig_twin/            # Fault simulation data
│   │   ├── exp_coolant/     # Coolant system faults
│   │   ├── exp_hydraulics/  # Hydraulic system faults
│   │   └── exp_probe/       # Probe system faults
│   ├── expert_graph/        # Expert-derived causal graph
│   └── categorical_encoding.json
├── src/                      # Source code for analysis and benchmarking
├── eval/                     # Evaluation scripts
│   ├── cd/                  # Causal discovery evaluation
│   └── rca/                 # Root cause analysis evaluation
```

## Dataset Access

The causRCA dataset is published on Zenodo:

**Dataset DOI:** [10.5281/zenodo.15876410](https://doi.org/10.5281/zenodo.15876410)

**Dataset Citation:**

```
Mehling, C. W., Pieper, S. and Lüke, T. (2025). causRCA: Real-World Dataset for 
Causal Discovery and Root Cause Analysis in Machinery (Version 1.0.0) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.15876410
```

## Installation

### Prerequisites
```bash
# Install system dependencies
sudo apt-get install gcc g++ make graphviz graphviz-dev
```

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate causrca

# Install package in development mode
pip install -e .
```

## Usage

### Dataset Description

Comprehensive dataset documentation is available in [data/README_DATASET.md](data/README_DATASET.md).

### Evaluation Scripts

- **Causal Discovery**: Benchmark causal discovery algorithms against expert knowledge
- **Root Cause Analysis**: Evaluate supervised and unsupervised RCA methods
- **Anomaly Detection**: Test fault detection capabilities

## Use Cases

- **Causal Discovery**: Benchmark learned causal graphs against expert-derived ground truth
- **Supervised Root Cause Analysis**: Train and test models on labeled fault diagnoses
- **Unsupervised Root Cause Analysis**: Identify manipulated variables with known ground truth
- **Anomaly Detection**: Evaluate fault detection in industrial time series

## Reproducing Paper Results

This section provides direct links between the evaluation scripts and the results tables / diagrams presented in our paper.

###  (Table 3) - Causal Discovery Results
```bash
# Generate Causal Discovery Performance Table for majority approach (dataset ensemble learning)
python eval/cd/cd_eval.py
```

###  (Table 4 & Table 5) - Supervised RCA Results and Unsupervised RCA Results
```bash
# Generate Supervised and Unsupervised RCA Performance Tables
python eval/rca/rca_eval.py
```

### (Table 6) - Causal RCA Performance using causal graphs obtained from different CD algorithms
```bash
# Generate Causal RCA Performance Table using causal graphs for unsupervised root cause analysis
# Evaluates AVG@3 performance scores across three datasets (Coolant, Hydraulic, Probe) 
# using causal graphs from four causal discovery algorithms compared to expert-graphs
python eval/rca/causal_rca_unsupervised_eval.py
```

### (Figure 2) - Boxplot for RCA performance with different F1 variants of expert graphs
```bash
# Generate Boxplot for RCA performance using different F1 variants of expert graphs
python eval/rca/causal_rca_f1_vars_eval.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.

## Authors & Acknowledgments

**Principal Investigator:**
- Carl Willy Mehling (Fraunhofer IWU) - [ORCID](https://orcid.org/0000-0002-0515-6800)

**Co-Investigators:**
- Sven Pieper (Fraunhofer IWU) - [ORCID](https://orcid.org/0000-0001-7436-8762)
- Tobias Lüke (Fraunhofer IWU) - [ORCID](https://orcid.org/0000-0002-5563-8779)

### Industry Partners

We gratefully acknowledge:
- **KAMAX Holding GmbH & Co. KG** - Real production data provision
- **Schuster Maschinenbau GmbH** - Digital twin development support
- **ISG Industrielle Steuerungstechnik GmbH** - Digital twin implementation
- **SEITEC GmbH** - Hardware-in-the-loop setup and OPC UA data recording

## Funding

This work was developed within the research project **KausaLAssist**, funded by the German Federal Ministry of Education and Research (BMBF) under grant 02P20A150.

## Related Publications

*Paper submission pending review - DOI will be added upon acceptance*

## Contributing

We welcome the use of the causRCA dataset for evaluating causal discovery and root cause analysis methods. We also encourage repository improvements through issues and pull requests.

[iwu-shield]: https://img.shields.io/badge/Fraunhofer-IWU-179C7D?style=flat-square
[github-shield]: https://img.shields.io/badge/github-%23121011.svg?style=flat-square&logo=github&logoColor=white
[apache2.0-licence]: https://img.shields.io/badge/License-Apache2.0-yellow.svg?style=flat-square
[doi-paper-shield]: https://img.shields.io/badge/DOI-tba.-blue.svg?style=flat-square