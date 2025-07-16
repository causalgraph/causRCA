
[![IWU][iwu-shield]](https://www.iwu.fraunhofer.de/)
[![zenodo-shield]](https://doi.org/10.5281/zenodo.15876410)
[![License][apache2.0-licence]](https://opensource.org/license/apache-2-0)
[![GitHub][github-shield]](https://github.com/causalgraph/causRCA)

# causRCA: Real-World Dataset for Causal Discovery and Root Cause Analysis in Machinery
`README in original version generated on 2025-07-13 by Carl Willy Mehling, Fraunhofer IWU. Up-to-date version on Zenodo (LINK)`

causRCA is a collection of time series datasets recorded from the CNC control of an industrial vertical lathe. 

The datasets comprise real-world recordings from normal factory operation and labeled fault data from a hardware-in-the-loop simulation. The fault datasets come with labels for the underlying (simulated) cause of the failure, a labeled diagnosis, and a causal model of all variables in the datasets. 

The extensive metadata and provided ground truth causal structure enable benchmarking of methods in causal discovery, root cause analysis, anomaly detection, and fault diagnosis in general.


## Use Cases & Applications

- **Causal Discovery**: Benchmark learned causal graphs against an expert-derived causal graph.
- **Supervised Root Cause Analysis**: Train and test models on labeled diagnosis for different fault scenarios.
- **Unsupervised Root Cause Analysis**: Identify manipulated variables in different fault scenarios with known ground truth.

---

## GENERAL INFORMATION

**Title of Dataset:** causRCA - Real-World Dataset for Causal Discovery and Root Cause Analysis in Machinery

### Author/Principal Investigator Information

- **Name:** Carl Willy Mehling
- **ORCID:** https://orcid.org/0000-0002-0515-6800
- **Institution:** Fraunhofer Institute for Machine Tools and Forming Technology IWU
- **Address:** 01187 Dresden, Germany

#### Co-Investigator 1

- **Name:** Sven Pieper
- **ORCID:** https://orcid.org/0000-0001-7436-8762
- **Institution:** Fraunhofer Institute for Machine Tools and Forming Technology IWU

#### Co-Investigator 2

- **Name:** Tobias Lüke
- **ORCID:** https://orcid.org/0000-0002-5563-8779
- **Institution:** Fraunhofer Institute for Machine Tools and Forming Technology IWU


**Date of data collection:** 2024-07-08 to 2025-02-28

**Geographic location of data collection:** Homberg (Ohm), Hessen, Germany and Erfurt, Thuringia, Germany

---

## SHARING & ACCESS INFORMATION

- **License:** Apache License 2.0
- **Recommended citation:**
  > Mehling, C. W., Pieper, S. and Lüke, T. (2025). (2025). *causRCA: Real-World Dataset for Causal Discovery and Root Cause Analysis in Machinery* (Version 1.0.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.15876410](https://doi.org/10.5281/zenodo.15876410)
- **Related publications:** TODO-Link CirpE Paper

---

## DATA & FILE OVERVIEW

### Directory structure

```plaintext
data/
 ┣ real_op/                    
 ┣ dig_twin/
 ┃ ┣ exp_coolant/
 ┃ ┣ exp_hydraulics/
 ┃ ┗ exp_probe/
 ┗ expert_graph/
```
The zipped data folder contains:
- **real_op/**: CSV files with time series data from normal operation.
- **dig_twin/**: Data from the digital twin experiments. Each group (coolant,hydraulics,probe) contains a causal subgraph as ground truth, different fault scenarios and multiple runs per scenario:
  - **exp_coolant/**: Coolant system faults 
  - **exp_hydraulics/**: Hydraulic system faults
  - **exp_probe/**: Probe system faults
- **expert_graph/**: GML and interactive HTML file with the expert-derived causal graph and lists of nodes and edges.

### Datasets summary
 (Sub-)graph         | #Nodes | #Edges| #Datasets Normal | #Datasets Fault | #Fault Scenarios | #Different Diagnoses | #Causing Variables |
| :----------        | :----  | :---- | :--------        | :---------------| :----------------| :---------           | :------------      |
| Lathe (Full graph) | 92     | 104   | 170              | 100             | 19               | 10                   | 14                 |
| --Probe            | 11     | 15    | 170              | 34              | 6                | 3                    | 2                  |
| --Hydraulics       | 17     | 18    | 170              | 41              | 9                | 5                    | 6                  |
| --Coolant          | 15     | 10    | 170              | 25              | 4                | 2                    | 6                  |
| --(Other Vars)     | 49     | 61    | 170              | -               | -                | -                    | -                  |

  *datasets from normal operation contain all machine variables and therefore all subgraphs and their respective variables within it. 

---

## METHODOLOGICAL INFORMATION

### Real Operation Data (`real_op`)

Data were recorded through an OPC UA interface during normal production cycles on a vertical lathe. These files capture baseline machine behavior under standard operating conditions, without induced or known faults.

### Digital Twin Data (`dig_twin`)

A hardware-in-the-loop digital twin was developed by connecting the original machine controller to a real-time simulation. Faults (e.g., valve leaks, filter clogs) were injected by manipulating specific twin variables, providing known ground-truth causes. Data were recorded via the same OPC UA interface to ensure consistent structure.

### Known limitations
Data was sampled via an OPC UA interface. The timestamps only reflect the published time of value change by the CNC and do not necessarily reflect the exact time of value changes. 

Consequently, the chronological order of changes across different variables is not strictly guaranteed. This may impact time-series analyses that are highly sensitive to precise temporal ordering.

---

## METHODS FOR PROCESSING

- see [causRCA Github Repository](https://github.com/causalgraph/causRCA)

---

## LICENSE & PERMISSIONS

- Data released under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## FUNDING

This work was developed within the research project KausaLAssist, funded on grant 02P20A150 by the German Federal Ministry of Education and Research (BMBF).

## ACKNOWLEDGEMENTS

The authors gratefully acknowledge the contributions of:

- **KAMAX Holding GmbH & Co. KG** for providing real production data from the vertical lathe.
- **Schuster Maschinenbau GmbH** for supporting the digital twin development with knowledge and the PLC project.
- **ISG Industrielle Steuerungstechnik GmbH** for developing the digital twin implementation.
- **SEITEC GmbH** for hosting the hardware-in-the-loop setup and developing the OPC UA data recording solution.

---

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of the dataset, the author(s) used generative AI tools to enhance the dataset's applicability by structuring data in an accessible format with extensive metadata, assist in coding transformations, and draft description content. All AI-generated output was reviewed and edited under human oversight, and no original dataset content was created by AI. 


<!-- MARKDOWN LINKS & IMAGES -->
[iwu-shield]: https://img.shields.io/badge/Fraunhofer-IWU-179C7D?style=flat-square
[github-shield]: https://img.shields.io/badge/github-%23121011.svg?style=flat-square&logo=github&logoColor=white
[apache2.0-licence]: https://img.shields.io/badge/License-Apache2.0-yellow.svg?style=flat-square
[zenodo-shield]: https://img.shields.io/badge/DOI-10.5281/zenodo.15876410-blue.svg?style=flat-square