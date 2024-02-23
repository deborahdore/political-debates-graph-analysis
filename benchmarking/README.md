<div align="center">

# BENCHMARKING

</div>

### DESCRIPTION

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
the [basic](output%2Fbasic) folder and in the corresponding [data.xlsx](output%2Fdata.xlsx).

However, our goal remains evaluating only three specific types of relations. Results of the models trained with
different kind of information but evaluated only on _support, attack and equivalence_ are available
at [special](output%2Fspecial)
and [pretrained](output%2Fpretrained).
The results in [special](output%2Fspecial) are obtained using no other addition other than the information already
present in
the dataset while the results in [pretrained](output%2Fpretrained) include the use of pretrained embeddings for the
arguments.

### EXECUTE AN EXPERIMENT

To execute the script, it's mandatory to run the [main](main.py) file and include some arguments (other hyperparameters
can be changed from the [config.py](config.py) file:

|        **ARGUMENT**         | **REQUIRED** | **DEFAULT** |                                                              **HELP**                                                              |
|:---------------------------:|:------------:|:-----------:|:----------------------------------------------------------------------------------------------------------------------------------:|
|         --generate          |      No      |    False    |                                Whether or not to generate a new dataset with the new configurations                                |
|         --optimize          |      No      |    False    |                                Whether or not to perform hyper-parameter optimization on the model                                 |
| --special_benchmarking_flag |      No      |    False    |                      Whether or not to perform evaluation with only support, attack and equivalent relations                       |
| --use_pretrained_embeddings |      No      |    False    |                                   Whether or not to use pretrained embeddings for the KGE models                                   |
|           --wandb           |      No      |    False    |                                        Whether or not to use Wandb to log model's training                                         |
|    --wandb_project_name     |      No      |    None     |                                                        Wandb's project name                                                        |
|           --model           |     Yes      |             |                  Model's name to train. Available: TransE, TransH, TransD, DistMult, RESCAL, HolE, ConvE, ConvKB                   |
|           --noise           |     Yes      |             |                             Noise in the dataset to train the model on. Available: 0, 10, 20, 30, 100                              |
|         --mode_text         |     Yes      |             |                           Configuration for the feature of the nodes. Available: "text" or "text+claim"                            |
|         --mode_node         |     Yes      |             | Types of relationships between nodes separated by a comma (can be more than one). Available: "claim+premise", "speaker" and "year" |
|      --output_dir_name      |     Yes      |             |                                                    Name of the output directory                                                    |

If you are not sure about node types and text types, take a look at this [explaination.pdf](output%2Fexplaination.pdf).

_Example_: <br>
`python main.py --model TransE --noise 0 --mode_text text --mode_node "speaker, year" --generate --optimize --special_benchmarking_flag --use_pretrained_embeddings`