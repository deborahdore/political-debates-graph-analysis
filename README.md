# political-debates-graph-analysis

The objective of this project is to implement knowledge embedding graph techniques on a dataset containing 41
presidential election debates in the United States spanning from 1960 to 2016. This dataset has been meticulously
annotated, with a specific focus on dissecting argument components (referred to as _premises_ and _claims_) and
delineating argument relations (namely, _support_ , _attack_ and _equivalence_).
This dataset can be explored with the tool DISPUTool publicly available [here](https://3ia-demos.inria.fr/disputool/).
More information about the dataset can be found in
this [GitHub repo](https://github.com/ElecDeb60To16/Dataset/tree/master).

Previously, the classification of argument relations, including support, attack, and equivalence, primarily relied on
deep learning techniques. This project endeavors to explore an innovative approach by considering the structural
attributes of constructed graphs to enhance the automated classification of argument relations.

## EXPLORATION DATA ANALYSIS

All the code for the exploration data analysis of the dataset can be found in the corresponding folder:
`visualization/graph_visualization.ipynb`

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

_Example graph of debate between Gore and Kemp_:
<img src="imgs/09-10-1996_debate_graph.svg">

## BENCHMARKING

An explaination of the work and the corresponding code can be found in the folder [benchmarking](benchmarking).

### REPRODUCIBILITY

* Python 3.9
* Nvidia V100 32 Gb
* [requirements.py](requirements.txt)

## ACKNOWLEDGEMENT

This work was supported by the French government, through the UCAJEDI Investments in the Future project managed by the
National Research Agency (ANR) under reference number ANR-15-IDEX-01. The authors are grateful to the OPAL
infrastructure and the Université Côte d’Azur’s Center for High-Performance Computing for providing resources and
support.

## LICENSE

Our code and data are licensed with a [License](LICENSE).
