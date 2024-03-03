import multiprocessing
import subprocess
import signal
import sys, os

subprocesses = dict()


def worker(file):
    subprocesses[file] = subprocess.Popen(["python", file])


if __name__ == "__main__":
    files = [
        "./scripts/sonification_main.py",
        "./scripts/sonification_input_module.py",
        "./scripts/sonification_visualize_module.py",
        # "./scripts/sonification_visualization_module.py",
    ]
    prcesses = dict()

    for i in files:
        prcesses[i] = multiprocessing.Process(target=worker(i))
        prcesses[i].start()

    while True:
        try:  # Wait for the processes to finish
            for i in files:
                prcesses[i].join()
        except KeyboardInterrupt:
            print("Terminating!")
            for subproc in subprocesses.items():
                subproc.terminate()
                subproc.kill()
            os.kill(os.getpid(), signal.SIGKILL)
