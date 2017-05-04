from collections import namedtuple


""" A container for transitions. """
Transition = namedtuple('Transition',
                        ('id', 'state', 'action', 'reward', 'state_', 'done'))
