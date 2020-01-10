import akro
import numpy as np
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.envs.env_spec import EnvSpec
from garage.envs.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.experiment import LocalRunner, run_experiment
from garage.sampler import InPlaceSampler
from garage.torch.algos import PEARLSAC
from garage.torch.embeddings import RecurrentEncoder
from garage.torch.modules import FlattenMLP, MLPEncoder
from garage.torch.policies import ContextConditionedPolicy, \
    TanhGaussianMLPPolicy, TanhGaussianMLPPolicy2
import garage.torch.utils as tu

params = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=50, # number of data sampling / training iterates
        num_initial_steps=20, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=4, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=4, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=20, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=6,  # nuumber of transitions to eval on
        batch_size=4, # number of transitions in the RL batch
        embedding_batch_size=4, # number of transitions in the context batch
        embedding_mini_batch_size=4, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=5, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=False, # False makes latent context deterministic
        use_next_obs_in_context=True, # use next obs if it is useful in distinguishing tasks
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # docker is not yet supported
    )
)


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    # create multi-task environment and sample tasks
    env = GarageEnv(normalize(HalfCheetahDirEnv()))
    runner = LocalRunner(snapshot_config)
    tasks = [0, 1]
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = params['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim \
        if params['algo_params']['use_next_obs_in_context'] \
            else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if params['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = params['net_size']
    recurrent = params['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MLPEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_dim=context_encoder_input_dim,
        output_dim=context_encoder_output_dim,
    )
    qf1 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + action_dim + latent_dim,
        output_dim=1,
    )
    qf2 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + action_dim + latent_dim,
        output_dim=1,
    )
    vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_dim=obs_dim + latent_dim,
        output_dim=1,
    )

    latent_space = akro.Box(low=-10000,
                            high=10000,
                            shape=(latent_dim, ),
                            dtype=np.float32)

    augmented_space = akro.Tuple(
            (env.observation_space, latent_space))
    augmented_env = EnvSpec(augmented_space,
                            env.action_space)

    policy = TanhGaussianMLPPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    
    agent = ContextConditionedPolicy(
        latent_dim=latent_dim,
        context_encoder=context_encoder,
        policy=policy,
        use_ib=params['algo_params']['use_information_bottleneck'],
        use_next_obs=params['algo_params']['use_next_obs_in_context'],
    )

    pearlsac = PEARLSAC(
        env=env,
        train_tasks=list(tasks[:params['n_train_tasks']]),
        eval_tasks=list(tasks[-params['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **params['algo_params']
    )

    runner.setup(algo=pearlsac, env=env, sampler_cls=InPlaceSampler,
        sampler_args=dict(max_path_length=params['algo_params']['max_path_length']))
    runner.train(n_epochs=500, batch_size=100)

tu.set_gpu_mode(False)

run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)