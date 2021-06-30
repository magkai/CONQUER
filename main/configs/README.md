Config Parameters
============

In the following, the used parameters for training and evaluation configurations are explained. Precomputed data is used for efficiency. We provide sample configs for our main experiments.

## Training Parameters ##

|Name | Used Values   |   Description |
| ---- | ------------ | ------------|
|seed | [12345, ..., 12349] | Seed values used in CONQUER to enable deterministic behavior for probabilistic components (network initialization, action sampling, etc.)  |
|entropy_const | 0.1 | Weight for entropy regularization term |
|learning_rate | 1e-3 | Initial learning rate for adam optimizer |
|num_episodes | 50 | Number of episodes (training samples) before updating the policy network |
|num_rollouts | 20 | Number of rollouts (= number of actions sampled per step) |
|num_epochs | 10 | Number of epochs the model should be trained (checkpoints are stored after each epoch) |
|ref_prediction    | [true, false] | Whether the (noisy) reformulation predictor should be used |
|real_user   |  [true, false]| Whether a noisy user model should be used |
|fractional_reward  |  [true, false] |  Whether reformulation probabilities should be directly used as reward  |
|negative_reward |    [true, false] | Whether reward of -1 should be used in case of an reformulation (optional, default: true) |
|observation_spec_shape_x |  [1001,1002] | Observation shape (first dimension): Size of concatenation of question (+history) + action embeddings (1002 if history is used) |
|observation_spec_shape_y |   769 |  Observation shape (second dimension): Size of concatenation of observation and action mask  |
|checkpoint_path |  -  | Storing location for model checkpoints  |
| pretrained |  [true, false] | Whether a pre-trained model should be loaded from which training will be continued (optional, default: false) |
|pretrained_path |  -  | Location from which pretrained model should be loaded (optional, used if "pretrained" is true)  |
|starts_per_question |   -  | Startpoints for RL walks determined upfront for each question in the trainset  |
|question_list |   - | List with all question ids |
|paths |   - | Paths per startpoint |
|answers |  - | All gold answers used in user simulator |
|questions|   - | All training questions |
|action_numbers |  - | Number of actions (paths) available from each startpoint |
|next_questions | - | Next information need per question  |
|next_reformulations |   - | Next available reformulation for each question |
|next_info_predictions | - | When noisy reformulation predictor is used, prediction probabilities that two questions have different info need (optional, if ref_prediction is true) |
|ref_predictions |   - |  When noisy reformulation predictor is used: prediction probabilities that two questions are reformulations (optional, if ref_prediction is true)  |
|next_info_fraction |   - | If fractional reward should be used: use probabilities from reformulation predictor directly (likelihood for different info need) (optional, if fractional_reward is true)|
|ref_fraction | - | If fractional reward should be used: use probabilities from reformulation predictor directly (likelihood for same info need)  (optional, if fractional_reward is true)|
|q_start_indices |   - |  Indices of question and startpoint ids |
|encoded_questions |   - | Question embeddings |
|encoded_actions |  - | Stacked action embeddings |
|encoded_history |   - | Location of encoded conversation history if used (optional, default: null)|



## Evaluation Parameters ##

|Name | Used Values   |   Description |
| ---- | ------------ | ------------|
|seed | [12345, ..., 12349] | Seed values used in CONQUER to enable deterministic behavior for probabilistic components (network initialization, action sampling, etc.)  |
|nbr_sample_actions | 5 | Number of top actions taken per agent for ranking |
| agg_type  |  ["add", "max", "majo", "maxmajo"] | Aggregation type for final ranking (optional, default: "add") |
|observation_spec_shape_x |  [1001,1002] | Observation shape (first dimension): Size of concatenation of question (+history) + action embeddings (1002 if history is used)  |
|observation_spec_shape_y |   769 |  Observation shape (second dimension): Size of concatenation of observation and action mask  |
|filename |   -  | Path were results should be stored  |
|checkpoint_path |  -  | Checkpoint location of trained model |
|checkpoint_nbr | 10 | Number of the model checkpoint to be loaded |
|startpoints | - | Startpoints for RL walks determined upfront for each question in the eval dataset |
|contextPaths | - | Paths available from each startpoint |
| action_nbrs  |  - |  Number of actions (paths) available from each startpoint |
|labels_dict | - | Dictionary containing the respective label for each Wikidata id  |
|bert_questions    | - | Question embeddings  |
|bert_actions   | - | Action embeddings  |
|bert_history |  - | Location of encoded conversation history if used (optional, default: null)|


**Parameter settings for specific experiments:**

For our experiments with different conversation histories, set ``encoded_history`` to 
1. ``null`` to not include any history (default),
2. ``../data/train_data/encoded_firstQuestions_trainset.pickle`` to use the first question in the conversation,
3. ``../data/train_data/encoded_firstQprevQAverage_trainset.pickle`` to use the average of the first and the previous questions in the conversation,
4. ``../data/train_data/encoded_firstQprevQ_RefAverage_trainset.pickle`` to use the average of all reformulations of the first intent and all reformulations of the previous intent as history

and set ``bert_history`` analogously to the respective data for the testset.

For our experiments with different action choices, set ``encoded_actions`` to
1. ``../data/train_data/encoded_paths_trainset.pickle`` to use the paths as actions (default),
2. ``../data/train_data/encoded_pathEnds_trainset.pickle`` to use the path + the destination entity (= answer entity),
3. ``../data/train_data/encoded_startPaths_trainset.pickle`` to use the context entity (= start entity) + path,
4. ``../data/train_data/encoded_facts_trainset.pickle`` to use the entire fact (start entity+path+destination entity) as actions

and set ``bert_actions`` analogously to the respective data for the testset.


For our experiments with different answer aggregation, set ``agg_type`` in the evaluation script to
1. ``add`` to add prediction scores of several agents if they arrive at the same entity (default),
2. ``max`` to use the higher score if several agents arrive at the same entity,
3. ``maxmajo`` to sort scores by the prediction score as main criterion and by majority voting (how many agent arrive at same entity) as second criterion (in case of ties),
4. ``majo`` to sort scores by majority voting  as main criterion and the prediction score as secondary criterion (in case of ties).
