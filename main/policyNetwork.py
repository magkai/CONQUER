import tensorflow as tf
from tf_agents.networks import network
from tf_agents.distributions import masked
import tensorflow_probability as tfp
from tf_agents.networks import utils
from tf_agents.utils import nest_utils


"""CONQUER policy network"""
class KGActionDistNet(network.DistributionNetwork):

    def __init__(self,

        seed_value,
        input_tensor_spec,
        output_tensor_spec,
        batch_squash=True,
        dtype=tf.float32,
        name='KGActionDistNet'
        ):
        
        self._output_tensor_spec = output_tensor_spec
     
        super(KGActionDistNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),    
            output_spec= tfp.distributions.Categorical,
            name=name)
          

        #get same initial values for same seed to make results reproducible
        initializer1 = tf.keras.initializers.GlorotUniform(seed=seed_value)
        initializer2 = tf.keras.initializers.GlorotUniform(seed=(seed_value+1))
        #define network
        self.dense1 = tf.keras.layers.Dense(768, activation=tf.nn.relu, name="dense1",kernel_initializer=initializer1)
        self.dense2 = tf.keras.layers.Dense(768, name="dense2", kernel_initializer=initializer2)
     
        self.dist = 0
        self.logits = 0
    

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec


    def get_distribution(self):
        return self.dist

    def get_logits(self):
        return self.logits


   
    def call(self,
           observations,
           step_type,
           network_state,
           training=False,
           mask=None):

      """get prediction from policy network
        this is called for collecting experience to get the distribution the agent can sample from 
        and called once again to get the distribution for a given time step when calculating the loss"""  

      is_empty = tf.equal(tf.size(observations), 0)
      if is_empty:
        return 0, network_state

      #outer rank will be 0 for one observation, if we have several for calculating the loss it is greater than 1
      outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
      #this is needed because a dense layer expects a batch dimension 
      batch_squash = utils.BatchSquash(outer_rank)
      observations = tf.nest.map_structure(batch_squash.flatten, observations)
      #get individual parts of observation: action mask, optional history embeddings, question and actions
      observations, mask = tf.split(observations, [768, 1],2)
      if observations.shape[1] == 1001:
        with_history = False
        observations, actions = tf.split(observations, [1, 1000], 1) #2x[batchsize,1,768], 1x[batchsize,1000,768]
      elif  observations.shape[1] == 1002:
        with_history = True
        history, question, actions = tf.split(observations, [1, 1, 1000], 1) #3x[batchsize,1,768], 1x[batchsize,1000,768]
    
      if with_history:
        observations = tf.keras.layers.concatenate([history, question], axis=2) #[batchsize, 1536]

      observations = tf.squeeze(observations, axis=1)
      availableActions = tf.transpose(actions, perm=[0, 2, 1])#[batchsize,768, 1000]

      x = self.dense1(observations)
      out = self.dense2(x) #[1,768]
      out = tf.expand_dims(out, -1) 
      #we multiply actions and output of network and get a matrix where each column is vector for one action, we sum over each column to get score for each action
      scores = tf.reduce_sum(tf.multiply(availableActions, out),1)#[batchsize,1000,1]
      self.logits = scores
      #prepare the mask
      mask = tf.squeeze(mask)
      mask_zero = tf.zeros_like(mask)#(scores>0 and scores<0)
      mask = tf.math.not_equal(mask, mask_zero)
      mask = tf.transpose(mask)
      if with_history:
        mask = mask[:-2]
      else:
          mask = mask[:-1]
      mask = tf.transpose(mask)
      #we convert it to categorical distribution, an action will be sampled from it
      #we use a masking distribution here because we can have less than 1000 valid actions, invalid ones are masked out
      self.dist = masked.MaskedCategorical(logits=scores, mask=mask)
      return self.dist, network_state
      