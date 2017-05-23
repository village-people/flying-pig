# Village People, 2017

from argparse import ArgumentParser

def get_args():
    args = ArgumentParser('PigChaseExperiment')
    args.add_argument('-t', '--type', type=str, default='random',
                      choices=['dqn', 'empathetic', 'astar', 'random'],
                      help='The type of baseline to run.')
    args.add_argument('-e', '--epochs', type=int, default=5,
                      help='Number of epochs to run.')
    args.add_argument('-es', '--epoch_steps', type=int, default=100,
                      help='Max no of steps per epoch.')
    args.add_argument('endpoints', nargs='*',
                      default=['127.0.0.1:10000', '127.0.0.1:10001'],
                      help='Minecraft client endpoints (ip(:port)?)+')
    return args.parse_args()


def parse_clients_args(endpoints):
    """ Return an array of tuples (ip, port) extracted from ip:port string
        :param args_clients:
        :return:
    """
    return [str.split(str(client), ':') for client in endpoints]


def visualize_training(visualizer, step, rewards, tag='Training'):
    visualizer.add_entry(step, '%s/reward per episode' % tag, sum(rewards))
    visualizer.add_entry(step, '%s/max.reward' % tag, max(rewards))
    visualizer.add_entry(step, '%s/min.reward' % tag, min(rewards))
    visualizer.add_entry(step, '%s/actions per episode' % tag, len(rewards) - 1)
