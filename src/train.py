#!/usr/bin/env python
import os
import pickle
import rnad
from tqdm import trange
import numpy as np
from sacred import Experiment
import wandb
import jax
import re

ex = Experiment()



@ex.config
def config():
    num_digits = 3
    hand_length = 3
    training_steps = 1000
    checkpoint_frequency = 100
    load_checkpoint = None
    save_dir = 'checkpoints'
    rnad_config = {
        "batch_size": 256,
        # Can pass in other kwargs here
    }


def checkpoint(agent, save_dir, training_step):
    '''Save the agent periodically.'''
    # Make a save directory if it does not exist.
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # for memory leak
    jax.clear_caches()

    # Save the agent in an output file.
    output_file = (
        save_dir + '/agent_{}.pickle'.format(training_step)
    )
    with open(output_file, 'wb') as f:
        pickle.dump(agent, f)


def train(
        num_digits,
        hand_length,
        training_steps,
        rnad_config,
        checkpoint_frequency=100,
        load_checkpoint=None,
        save_dir='checkpoints',
):
    '''Train an agent to play a game.'''
    if load_checkpoint is None:
        last_step = -1
        # Create agent
        agent = rnad.RNaDSolver(
           rnad.RNaDConfig(
                game_name='python_liars_poker(num_digits={},hand_length={})'.format(
                    num_digits, hand_length
                ),
               **rnad_config
           )
        )
    else:
        m = re.search(r"agent_(\d+)\.pickle", load_checkpoint)
        if m:
            last_step = int(m.group(1))
        else:
            raise RuntimeError('unable to parse step from filename %d' % load_checkpoint)

        with open(load_checkpoint, 'rb') as f:
            agent = pickle.load(f)
        # todo(ciamac): check that num_digits and hand_length match

    # Train agent for 'training_steps' iterations
    losses = []
    for step in trange(last_step + 1, last_step + training_steps + 1):
      logs = agent.step()
      losses.append(logs['loss'])
      wandb.log({"loss": logs['loss']})

      if step % checkpoint_frequency == checkpoint_frequency - 1:
        print('Step: {}; Loss: {}'.format(step, np.mean(losses)))
        checkpoint(agent, save_dir, step)

    # Save final model
    print('Step: {}'.format(step))
    checkpoint(agent, save_dir, step)


@ex.automain
def main(num_digits,
         hand_length,
         training_steps,
         checkpoint_frequency,
         save_dir,
         load_checkpoint,
         rnad_config,
         id,
         _config):
    """
    :param num_digits: Number of digits to include from the range 1, 2, ..., 9, 0 in player hands.
    :param hand_length: Number of digits in each player's hand.
    :param training_steps: Number of steps to train agent.
    :param checkpoint_frequency: How often (in training steps) to save the agent.
    :param save_dir: Where to save the agent.
    """

    job_id = os.environ.get("JOB_ID", "local")
    print("Job ID:", job_id)

    wandb.init(
        project="liars-poker__train-rnad",
        entity="columbia_gsb",
        config=_config,
        name=id + "-" + job_id,
    )
    train(
        min(num_digits, 10),
        min(hand_length, 10),
        training_steps,
        rnad_config,
        checkpoint_frequency,
        load_checkpoint,
        save_dir
    )
