import multiprocessing
import subprocess
import signal
import sys, os
import psutil

subprocesses = dict()


def worker(file):
    subprocesses[file] = subprocess.Popen(["python", file, "0"])


if __name__ == "__main__":
    files = [
        "./scripts/sonification_main.py",
        "./scripts/sonification_input_module.py",
        "./scripts/sonification_visualize_module.py",
        # "./scripts/sonification_inference_module.py",
    ]
    # Check if there is a process running with name containing 'python'
    for process in psutil.process_iter(["name"]):
        if "python" in process.info["name"]:
            # Skip the current process
            if process.pid == os.getpid():
                continue
            # Kill the process
            process.kill()

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
