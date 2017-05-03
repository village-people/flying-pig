""" Malmo Challenge Entry Point.
"""

from threading import Thread, active_count
from time import sleep

from agents.workers import run_alien_agent, train_our_agent
from utils.utils import get_args


if __name__ == "__main__":

    args = get_args()

    threads = [
        # alien's thread
        Thread(target=run_alien_agent, args=("Agent_1", args.endpoints, 0)),

        # our agent's thread
        Thread(target=train_our_agent, args=("Agent_2", args.endpoints, 1,
                                             False, args))
    ]

    for t in threads:
        t.daemon = True
        t.start()
        sleep(1)    # apparently the server needs some time

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.001)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')
