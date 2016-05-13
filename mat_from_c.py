import atexit
import binascii
from subprocess import Popen, PIPE
import os
import os.path
import sys

import numpy as np
import cv2

class CMatProcess:
    def __init__(self, processPath='./cmatprocess/cmatprocess', nCores=4):
        assert processPath is not None

        self.cmd = [processPath, str(nCores)]

        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE,
                       stderr=PIPE, bufsize=0)

        def exitHandler():
            if self.p.poll() is None:
                self.p.kill()
        atexit.register(exitHandler)

    def processPaths(self, imgPaths, f):
        rc = self.p.poll()
        if rc is not None and rc != 0:
            raise Exception("Error from process!")

        self.p.stdin.write(f + ' ' + ' '.join(imgPaths) + '\n')
        try:
            n = int(self.p.stdout.readline())
            result = []
            for i in range(n):
                fileLocation = self.p.stdout.readline().strip()
                if len(fileLocation) > 3:
                    result.append(cv2.imread(fileLocation))

            return result
        except Exception as e:
            self.p.kill()
            stdout, stderr = self.p.communicate()
            print("Error from process output: " + stdout + '\n' + stderr)
            sys.exit(-1)