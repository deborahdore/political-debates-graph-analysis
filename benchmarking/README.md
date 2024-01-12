# BENCHMARKING

In this project we explore the possibility of using KGE models to predict relationships (_support, attack and
equivalent_)
between arguments in a graph taken from the political debates of US Presidents from 1960 to 2016. <br>

### FIRST APPROACH AND DATASET

The dataset is unbalanced, having _21689 support_ relations, _3835 attack_ relations and only _706 equivalent_ relations
over a total of _38667 different nodes/arguments_. <br>

An example of two connected argument is the following:<br>

- ARGUMENT 1: _I did not criticize him for not calling for free elections_
- RELATION: **ATTACK**
- ARGUMENT 2: _Senator Kennedy also indicated with regard to Cuba that he thought that I had made a mistake when I was
  in Cuba in not calling for free elections in that country_

A first approach was to train TransE, ConvE and DistMult on the dataset as is. Therefore, only having argument as nodes
and three relations as edges. However the results were poor. So an idea was to add information taken from the dataset
such as date of the debate, speakers, type of argument (claim/premise). The results of the evaluation can be found on
the [basic](results/basic) folder and in the corresponding [Excel sheet](results/results.xlsx).

However, our goal remains evaluating only three specific types of relations. Results of the models trained with
different
kind of informations but evaluated only on _support, attack and equivalence_ are available [here](results/special)
and [here](results/pretrained-special).
The first are the results with no other addition other than the information in the dataset while the seconds include the
use of pretrained embeddings for the arguments.

Some experiments were conducted to improve the results and evaluate them. It can be found [here](results/experiments).

### EXECUTE AN EXPERIMENT

To execute the script, it's mandatory to run the [main](main.py) file and include some arguments:

|        **ARGUMENT**         | **REQUIRED** | **DEFAULT** |                                                    **HELP**                                                    |
|:---------------------------:|:------------:|:-----------:|:--------------------------------------------------------------------------------------------------------------:|
|         --generate          |      No      |    False    |                      Whether or not to generate a new dataset with the new configurations                      |
|         --optimize          |      No      |    False    |                      Whether or not to perform hyper-parameter optimization on the model                       |
| --special_benchmarking_flag |      No      |    False    |            Whether or not to perform evaluation with only support, attack and equivalent relations             |
| --use_pretrained_embeddings |      No      |    False    |                         Whether or not to use pretrained embeddings for the KGE models                         |
|           --wandb           |      No      |    False    |                              Whether or not to use Wandb to log model's training                               |
|    --wandb_project_name     |      No      |    None     |                                              Wandb's project name                                              |
|           --model           |     Yes      |             | Model's name to train. Available: TransE, DistMult, ComplEx, HolE, ConvE, PairRE, AutoSF, RotatE, TransH, BoxE |
|           --noise           |     Yes      |             |                   Noise in the dataset to train the model on. Available: 0, 10, 20, 30, 100                    |
|         --mode_text         |     Yes      |             |                 Configuration for the feature of the nodes. Available: "text" or "text+claim"                  |
|         --mode_node         |     Yes      |             | Types of relationships between nodes (can be more than one). Available: "claim+premise", "speaker" and "year"  |

If you are not sure about node types and text types, take a look at this [explanation](results/example_nodes.pdf).

_Example_: <br>
`python main.py --model TransE --noise 0 --mode_text text --mode_node "speaker, year" --generate --special_benchmarking_flag --use_pretrained_embeddings`