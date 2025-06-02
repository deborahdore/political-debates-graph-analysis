<div align="center">

# POLITICAL DEBATES GRAPH ANALYSIS

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white&labelColor=gray"></a>
<a href="https://github.com/pykeen/pykeen?tab=readme-ov-file#citation"><img alt="PyKEEN" src="https://img.shields.io/badge/PyKEEN-blue?logo=github&style=flat&labelColor=gray"></a>
<a href="https://github.com/pierpaologoffredo/ElecDeb60to20"><img alt="Dataset" src="https://img.shields.io/badge/ElecDeb60to20-green?logo=github&style=flat&labelColor=gray"></a><br>

</div>
This repo contains the code for the paper _Leveraging Graph Structural Knowledge to Improve Argument Relation Prediction in Political Debates_ accepted at the 12th Workshop on Argument Mining @ ACL 2025 (Deborah Dore, Stefano Faralli, Serena Villata). 

## DESCRIPTION

The objective of this project is to implement knowledge embedding graph techniques on a dataset containing 44
presidential election debates in the United States spanning from 1960 to 2020. This dataset has been meticulously
annotated, with a specific focus on dissecting argument components (referred to as _premises_ and _claims_) and
delineating argument relations (namely, _support_ , _attack_ and _equivalence_).
This dataset can be explored with the tool DISPUTool publicly available [here](https://3ia-demos.inria.fr/disputool/).
More information about the dataset can be found in
this [GitHub repo](https://github.com/pierpaologoffredo/ElecDeb60to20).

Previously, the classification of argument relations, including support, attack, and equivalence, primarily relied on
deep learning techniques. This project endeavors to explore an innovative approach by considering the structural
attributes of constructed graphs to enhance the automated classification of argument relations.

## EXPLORATION DATA ANALYSIS

All the code for the exploration data analysis of the dataset can be found in the corresponding
folder: [`visualization/graph_visualization.ipynb`](notebooks%2Fgraph_visualization.ipynb)

| **Statistics**             | **num** | 
|:---------------------------|:-------:|
| Debates                    |   44    |
| Total speakers             |   64    |
| Edges                      |  26230  |
| Nodes                      |  38667  |
| Claim node                 |  25078  |
| Premise node               |  13589  |
| Support Nodes Relations    |  21689  |   
| Attack Nodes Relations     |  3835   |
| Equivalent Nodes Relations |   706   |

<a target="_blank" href="https://colab.research.google.com/github/deborahdore/political-debates-graph-analysis/blob/main/visualization/graph_visualization.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

| **Example graph of debate between Gore and Kemp**                              |
|:-------------------------------------------------------------------------------|
| ![09-10-1996_debate_graph.svg](notebooks%2Fimgs%2F09-10-1996_debate_graph.svg) |

## BENCHMARKING

An explaination of the work and the corresponding code can be found in the folder [benchmarking](benchmarking).

## REPRODUCIBILITY

* Python 3.8
* Nvidia V100 32 Gb
* [environment.yaml](environment.yaml)
* [Trained Models and Data](https://drive.google.com/drive/folders/1Sz9PSAemUFqKXDFWVYezeNNXt4brMmvl?usp=share_link)

## LICENSE

Our code and data are licensed with a [License](LICENSE).

## ACKNOWLEDGEMENT

This work was supported by the French government, through the UCAJEDI Investments in the Future project managed by the
National Research Agency (ANR) under reference number ANR-15-IDEX-01. The authors are grateful to the OPAL
infrastructure and the Université Côte d’Azur’s Center for High-Performance Computing for providing resources and
support.
