"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage.misc import special
from garage.tf.algos.batch_polopt2 import BatchPolopt2
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import center_advs
from garage.tf.misc.tensor_utils import compute_advantages_ragged
from garage.tf.misc.tensor_utils import discounted_returns_lengths
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.misc.tensor_utils import new_ragged_dense_tensor
from garage.tf.misc.tensor_utils import pad_tensor_to_max_len
from garage.tf.misc.tensor_utils import positive_advs
from garage.tf.optimizers import LbfgsOptimizer


class NPO2(BatchPolopt2):
    """Natural Policy Gradient Optimization.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        name (str): The name of the algorithm.

    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='vanilla',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 name='NPO'):
        self.name = name
        self._name_scope = tf.name_scope(self.name)
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._pg_loss = pg_loss
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = LbfgsOptimizer

        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')

        with self._name_scope:
            self.optimizer = optimizer(**optimizer_args)
            self.lr_clip_range = float(lr_clip_range)
            self.max_kl_step = float(max_kl_step)
            self.policy_ent_coeff = float(policy_ent_coeff)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         flatten_input=flatten_input)

    def init_opt(self):
        """Initialize optimizater."""
        pol_loss_inputs, pol_opt_inputs, path_lengths = self._build_inputs()
        self._policy_opt_inputs = pol_opt_inputs

        pol_loss, pol_kl = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(loss=pol_loss,
                                  target=self.policy,
                                  leq_constraint=(pol_kl, self.max_kl_step),
                                  inputs=flatten_inputs(
                                      self._policy_opt_inputs),
                                  extra_inputs=[path_lengths],
                                  constraint_name='mean_kl')
        return dict()

    def optimize_policy(self, itr, samples_data):
        """Optimize policy."""
        policy_opt_input_values, path_lengths = self._policy_opt_input_values(samples_data)
        # Train policy network
        logger.log('Computing loss before')
        loss_before = self.optimizer.loss(policy_opt_input_values, [path_lengths])
        logger.log('Computing KL before')
        policy_kl_before = self.f_policy_kl(*policy_opt_input_values + [path_lengths])
        logger.log('Optimizing')
        self.optimizer.optimize(policy_opt_input_values, extra_inputs=[path_lengths])
        logger.log('Computing KL after')
        policy_kl = self.f_policy_kl(*policy_opt_input_values + [path_lengths])
        logger.log('Computing loss after')
        loss_after = self.optimizer.loss(policy_opt_input_values, [path_lengths])
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = np.mean(self.f_policy_entropy(*policy_opt_input_values + [path_lengths]))
        tabular.record('{}/Entropy'.format(self.policy.name), pol_ent)
        tabular.record('{}/Perplexity'.format(self.policy.name), np.exp(pol_ent))

        self._fit_baseline(samples_data, path_lengths)
        self.old_policy.model.parameters = self.policy.model.parameters

    def get_itr_snapshot(self, itr):
        """Get iteration snapshot."""
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
        )

    def _build_inputs(self):
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        with tf.name_scope('inputs'):
            if self.flatten_input:
                obs_var = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=[None, observation_space.flat_dim],
                    name='obs')
            else:
                obs_var = observation_space.to_tf_placeholder(name='obs',
                                                          batch_dims=1)
            lengths = tf.compat.v1.placeholder(tf.int32,
                                               shape=(None,),
                                               name='lengths')
            masks = pad_tensor_to_max_len(lengths, self.max_path_length)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=1)
            reward_var = tf.compat.v1.placeholder(name='reward',
                                                  shape=[None],
                                                  dtype=tf.float32)
            baseline_var = tf.compat.v1.placeholder(name='baseline',
                                                    shape=[None],
                                                    dtype=tf.float32)
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

        if self.policy.recurrent:
            obs_input = obs_var
            # concat other input with obs_var to become the final state input
            if self.policy._state_include_action:
                for k in self.policy.state_info_keys:
                    extra_state_var = policy_state_info_vars[k]
                    extra_state_var = tf.cast(extra_state_var, tf.float32)
                    obs_input = tf.concat(axis=-1,
                                          values=[obs_input, extra_state_var])
            # [B, T, *obs_dims]
            state_ragged = new_ragged_dense_tensor(obs_input, lengths, self.max_path_length)
            # [B, T, *act_dims]
            action_input = new_ragged_dense_tensor(action_var, lengths, self.max_path_length)
            self.old_policy.build(state_ragged)
            self.policy.build(state_ragged)
        else:
            self.old_policy.build(obs_var)
            self.policy.build(obs_var)
            action_input = action_var

        # policy loss and optimizer inputs
        # All these input tensors are used to construct the loss
        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            action_var=action_input,
            reward_var=reward_var,
            baseline_var=baseline_var,
            length_var=lengths,
            masks=masks,
            policy_state_info_vars=policy_state_info_vars,
        )
        # all these input tensors are passed to optimizer and will be fed
        # into the operations built
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
        )

        return policy_loss_inputs, policy_opt_inputs, lengths

    def _build_policy_loss(self, i):
        pol_dist = self.policy.distribution
        # [B * (T)]
        policy_entropy = self._build_entropy_term(i)
        rewards = i.reward_var

        if self._maximum_entropy:
            with tf.name_scope('augmented_rewards'):
                rewards = i.reward_var + self.policy_ent_coeff * policy_entropy

        with tf.name_scope('policy_loss'):
            # [B * (T)]
            adv = compute_advantages_ragged(self.discount,
                                            self.gae_lambda,
                                            self.max_path_length,
                                            i.baseline_var,
                                            rewards,
                                            i.length_var,
                                            i.masks,
                                            name='adv')
            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                adv = center_advs(adv, axes=[0], eps=eps)

            if self.positive_adv:
                adv = positive_advs(adv, eps)

            # Calculate loss function and KL divergence
            with tf.name_scope('kl'):
                kl = self.old_policy.distribution.kl_divergence(
                    self.policy.distribution)
                if self.policy.recurrent:
                    kl = tf.boolean_mask(kl, i.masks)
                pol_mean_kl = tf.reduce_mean(kl)

            # Calculate vanilla loss
            with tf.name_scope('vanilla_loss'):
                ll = self.policy.distribution.log_prob(
                    i.action_var, name='log_likelihood')
                if self.policy.recurrent:
                    ll = tf.boolean_mask(ll, i.masks)
                vanilla = ll * adv

            # Calculate surrogate loss
            with tf.name_scope('surrogate_loss'):
                lr = tf.exp(self.policy.distribution.log_prob(
                    i.action_var) - tf.stop_gradient(
                        self.old_policy.distribution.log_prob(
                            i.action_var)))
                if self.policy.recurrent:
                    lr = tf.boolean_mask(lr, i.masks)
                surrogate = lr * adv

            # Finalize objective function
            with tf.name_scope('loss'):
                if self._pg_loss == 'vanilla':
                    # VPG uses the vanilla objective
                    obj = tf.identity(vanilla, name='vanilla_obj')
                elif self._pg_loss == 'surrogate':
                    # TRPO uses the standard surrogate objective
                    obj = tf.identity(surrogate, name='surr_obj')
                elif self._pg_loss == 'surrogate_clip':
                    lr_clip = tf.clip_by_value(lr,
                                               1 - self.lr_clip_range,
                                               1 + self.lr_clip_range,
                                               name='lr_clip')
                    surr_clip = lr_clip * adv
                    obj = tf.minimum(surrogate, surr_clip, name='surr_obj')
                if self._entropy_regularzied:
                    obj += self.policy_ent_coeff * policy_entropy

                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                loss = -tf.reduce_mean(obj)

            # Diagnostic functions
            self.f_policy_kl = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs) + [i.length_var], pol_mean_kl)

            self.f_rewards = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs) + [i.length_var], rewards)

            returns = discounted_returns_lengths(self.discount, self.max_path_length,
                      rewards, i.length_var, i.masks)

            self.f_returns = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs) + [i.length_var], returns)

            return loss, pol_mean_kl

    def _build_entropy_term(self, i):
        pol_dist = self.policy.distribution

        with tf.name_scope('policy_entropy'):
            if self._use_neg_logli_entropy:
                policy_entropy = -pol_dist.log_prob(i.action_var,
                        name='policy_log_likeli')
            else:
                policy_entropy = pol_dist.entropy()

            # This prevents entropy from becoming negative for small policy std
            if self._use_softplus_entropy:
                policy_entropy = tf.nn.softplus(policy_entropy)

            if self._stop_entropy_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

            if self.policy.recurrent:
                policy_entropy = tf.boolean_mask(policy_entropy, i.masks)

        self.f_policy_entropy = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs) + [i.length_var], policy_entropy)

        return policy_entropy

    def _fit_baseline(self, samples_data, path_lengths):
        """Update baselines from samples."""
        policy_opt_input_values, path_lengths = self._policy_opt_input_values(samples_data)

        # Augment reward from baselines
        augmented_reward = self.f_rewards(*policy_opt_input_values + [path_lengths])
        augmented_return = self.f_returns(*policy_opt_input_values + [path_lengths])

        paths = samples_data['paths']
        baselines = samples_data['baselines']

        samples_data['rewards'] = augmented_reward
        samples_data['returns'] = augmented_return

        # Calculate explained variance
        ev = special.explained_variance_1d(baselines,
                                           augmented_return)
        tabular.record('{}/ExplainedVariance'.format(self.baseline.name), ev)

        # Fit baseline
        logger.log('Fitting baseline...')
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

    def _policy_opt_input_values(self, samples_data):
        """Map rollout samples to the policy optimizer inputs."""
        policy_state_info_list = [
            samples_data['agent_infos'][k] for k in self.policy.state_info_keys
        ]

        # All data are of shape [B * (T), *dims]
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data['observations'],
            action_var=samples_data['actions'],
            reward_var=samples_data['rewards'],
            baseline_var=samples_data['baselines'],
            policy_state_info_vars_list=policy_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values), samples_data['lengths']

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration."""
        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')

    def __getstate__(self):
        """Get state."""
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_policy_opt_inputs']
        del data['f_policy_entropy']
        del data['f_policy_kl']
        del data['f_rewards']
        del data['f_returns']
        return data

    def __setstate__(self, state):
        """Set state."""
        self.__dict__ = state
        self._name_scope = tf.name_scope(self.name)
        self.init_opt()
