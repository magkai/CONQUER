CONQUER: Reinforcement Learning from Reformulations in Conversational QA over KGs
============

Description
------------

This repository contains the code and data for our SIGIR'21 full paper. In this paper, we present CONQUER, a reinforcement learning model that can learn from a conversational stream of questions and reformulations. A reformulation is likely to be triggered by an incorrect system response, whereas a new follow-up question could be a positive signal on the previous turn’s answer. CONQUER is trained via noisy rewards coming from the reformulation likelihoods.
The answering process is modeled as multiple agents walking in parallel on the knowledge graph: 

<center><img src="kg_graph.png"  alt="kg_graph" width=80%  /></center>


*KG excerpt required for answering "When was Avengers: Endgame released in Germany?" and "What was the next from Marvel?".
Agents are shown with possible walk directions. The colored box ("Spider-man: Far from Home") is the correct answer.*

For more details see our paper: [Reinforcement Learning from Reformulations in Conversational Question Answering over Knowledge Graphs](https://arxiv.org/abs/2105.04850)

If you use this code, please cite:
```bibtex
@inproceedings{kaiser2021reinforcement,
  title={Reinforcement Learning from Reformulations in Conversational Question Answering over Knowledge Graphs},
  author = {Kaiser, Magdalena and Saha Roy, Rishiraj and Weikum, Gerhard},
  booktitle={SIGIR},
  year={2021}
 }
```

Setup 
------

The following software is required:

* Python 3.7

* Spacy 2.1.6

* Numpy 1.20.1

* Tensorflow 2.20

* Transformers 3.5.1

* TF-Agents 0.5.0

* Neo4j 1.7.2

* Scikit-learn 0.21.2

To install the required libraries, it is recommended to create a virtual environment:

    python3 -m venv ENV_conquer
    source ENV_conquer/bin/activate
    pip install -r requirements.txt


Data
------
The benchmark and all required intermediate data can be downloaded from here (unzip and put it in the root folder of the cloned github repo): https://conquer.mpi-inf.mpg.de/static/data.zip 

        

Training CONQUER
------
Execute in the `main` directory:

    python rlMain.py configs/train_REFTYPE_USERTYPE_config.json

where ``REFTYPE`` can either be *idealRef* or *noisyRef* to select the ideal/noisy reformulation predictor 
and ``USERTYPE`` can either be *idealUser* or *noisyUser* to apply the ideal/noisy user model respectively

Further details about the config parameters can be found in `main/configs`. 
The provided config files use the pre-computed data (downloaded at the previous step). For creating the required data from scratch see **Running Data Preprocessing Steps** below. We also included the trained models for our main experiments in the provided data folder.

Evaluating CONQUER
------
Execute in the `main` directory:

    python rlEval.py configs/eval_REFTYPE_USERTYPE_config.json

where ``REFTYPE`` can either be *idealRef* or *noisyRef* to select the ideal/noisy reformulation predictor, ``USERTYPE`` can either be *idealUser* or *noisyUser* to apply the ideal/noisy user model respectively.

The produced output file consists of the following elements:
```
CONV_ID,QUESTION,ANSWER,GOLD_ANSWER,PRECISION@1, HITS@5, MRR
```
where ``CONV_ID`` is the respective id of the question in the dataset, followed by the user question, the system answer, the gold labeled answers from the dataset (small answer set, typicially one) and the three metrics P@1, Hits@5 and MRR. 
The question id is formated in the following way ``X-Y-Z``, where ``X`` is the number of the conversation, ``Y`` is the turn number and ``Z`` is the reformulation number (only present in case of a reformulation).
Here is an example output line: 
```
8961-0,What is the name of the writer of The Secret Garden?,Q276028,['Q276028'],1.0,1.0,1.0
```
At the end of the file, a summary with the average results on the three metrics, as well as further stats like the 
*total number of reformulations triggered* and the *number of questions answered correctly on the ith attempt* is provided.

Training & Evaluating Reformulation Predictor
-------
Execute in the `reformulation_prediction` directory:

1. The required data for fine-tuning BERT for reformulation prediction is part of the provided data folder (see download above). Alternatively, it can be created with the following script:

```python
   python createRefDataset.py 
```

2. Train the reformulation predictor (fine-tune BERT model):

```
   python finetuneRefPredictor.py
```
Note that we also provide the fine-tuned models in the data folder. 

3. Evaluate the performance of the reformulation predictor:
```
   python refPredictEval.py
```
The evaluation output consists of the following metrics:
```
loss, accuracy, f1, precision, recall
f1_0_labels, precision_0_labels, recall_0_labels
f1_1_labels, precision_1_labels, recall_1_labels
```
where ``0_labels`` refers to the classification of different information needs and ``1_labels`` to the classification of reformulations.

OPTIONAL: Running the Neo4j Database
------
For running the **Context Entity Detection** (see next step), we need to access our KG that we loaded into a Neo4j database. 
This database can be set up in the following way:
1. For using neo4j, JDK 11 or higher is required and can be downloaded from here: https://jdk.java.net/.

2. We used the neo4j-community-4.0.5 version in our experiments. The latest versions of neo4j can be downloaded from here: https://neo4j.com/download-center/#community

3. We are using the *apoc* library that provides additional functionality to neo4j. In newer versions of neo4j you can find the respective jar file inside the ``labs`` folder of the downloaded neo4j directory. Otherwise, you can download a compatible version here: https://neo4j.com/labs/apoc/4.1/installation/.
Move the apoc jar file (from ``labs``) to ``plugins`` and include the following line in ``configs/neo4j.conf``: ``dbms.security.procedures.unrestricted=apoc.*`` to be able to use the library. 

4. Load the CONQUER KG representation into the neo4j database. 

   a) You can find the required KG dumps here:  https://conquer.mpi-inf.mpg.de/static/dumps.zip. Unzip and put them into the ``data`` folder. Alternatively, you can build our KG representation from scratch (see **Running KG Preparation Steps** below).

   b) Execute in the ``kg_processing`` directory:
```
   bash loadKGIntoNeo4j.sh
```
The final database requires around 65 GB of disk space and around 30 GB of RAM is necessary to run the database.

5.  The database can be started with the following command:

```
   $NEO4J_HOME/bin/neo4j start
```
where $NEO4J_HOME is the path to the downloaded neo4j directory.

6. You need to change the default password:

```
   $NEO4J_HOME/bin/cypher-shell
```
and type in the defaults: *user: neo4j*, *password: neo4j*, then you are prompted to type in a new password. Restart the database (`` $NEO4J_HOME/bin/neo4j stop`` and then again `` $NEO4J_HOME/bin/neo4j start``).

OPTIONAL: Running Context Entity Detection
------
The context entities (= startpoints for the RL walk) along with their respective KG paths (= actions) have been pre-computed and can directly be used (see **Data** section above) for the RL. In case you want to re-run the context entity detection, the following is required:

1. We use ELQ as our NED tool. To make use of it, clone the following repo:
```
   git clone https://github.com/facebookresearch/BLINK.git
```  
   Place the BLINK directory inside the root directory of CONQUER and perform the setup steps described here: https://github.com/facebookresearch/BLINK/tree/master/elq


2. Access to our KG, which has been loaded into a neo4j database, is required. 
Perform the steps described in **Running the Neo4j Database** above for this.

3. Once the database is running, execute the following commands in the `context_entity_detection` directory:

```
   mkdir models/
   python contextEntityRetrieval.py
```

OPTIONAL: Running Data Preprocessing Steps
------
We provide the preprocessed data (see above) for easy usage. In case you want to build it from scratch, the following steps are necessary:

Execute in the `data_preprocessing` directory:

1. Process the benchmark data:

```python
   python processConvRefData.py 
```
2. Encode questions and actions with BERT:

```python
   python doBertEncoding.py
```

3. Encode the conversation history (in case you want to include it):

```python
   python encodeConversationHistory.py 
```

4. Make best use of the retrieved context entities (for training only):

```python
   python prepareStartPoints.py 
```

5. Pre-compute reformulation predictions (for training only):

```python
   python precomputeRefPredictions.py 
```

OPTIONAL: Running KG Preparation Steps
------

1. In this project, we use Wikidata as our Knowledge Graph. You can download the most recent wikidata dump here (around 2 TB): https://dumps.wikimedia.org/wikidatawiki/entities/

2. Our goal was to make the dump most suitable for QA by performing a sequence of filtering steps, which results in a much smaller dump (around 42 GB). 
Our wikidata processing pipeline is located in a separate project. Clone it into the ``kg_processing`` directory of CONQUER:
```
   git clone https://github.com/PhilippChr/wikidata-core-for-QA.git
```

   and execute inside the ``wikidata-core-for-QA`` directory:
```
   bash prepare_wikidata_for_qa.sh <wikidata_dump_path> <number_of_workers>
```
   where ``wikidata_dump_path`` is the location of the downloaded Wikidata dump and ``number_of_workers`` specifies the number of processes that should run in parallel.

3. In order to get the CONQUER specific KG representations (where the qualifier information is added to the paths of the main fact and vice versa), execute inside the ``kg_processing`` directory:
```
   bash createQualifierDumpRepresentation.sh
```

4. Finally, we need to load the CONQUER KG into the neo4j database for later access as described in **Running the Neo4j Database** above.

