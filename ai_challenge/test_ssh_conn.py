# Village People, 2017

from pexpect import pxssh
import time

PATH = 'Documents/Malmo-0.21.0-Linux-Ubuntu-16.04-64bit_withBoost/Minecraft/'

def get_session(ip, user, password):
    s = pxssh.pxssh()
    if not s.login(ip, user, password):
        print("SSH session failed on login.")
        print(str(s))
    else:
        print("SSH session login successful")
        # s.setwinsize(24, s.maxread)
    return s


def launch_clients(session, instances):
    session.sendline('cd ' + PATH)

    for i in instances:
        time.sleep(0.1)
        session.sendline(
            'xvfb-run -a -e /dev/stdout ./launchClient.sh -port 10000')

if __name__ == "__main__":
    targets = [('172.19.3.234', 'aimas', 'aimas@303')]
    sessions = []
    for c in targets:
        print(*c)
        sessions.append(get_session(*c))

    for s in sessions:
        launch_clients(s, [(10000, 10001)])