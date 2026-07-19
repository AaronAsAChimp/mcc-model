import json
from typing import Optional, TextIO


class ExtractsFile:
    _filename: str
    _fd: Optional[TextIO]

    def __init__(self, filename):
        self._filename = filename
        self._fd = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._fd is None:
            raise Exception("File not open")

        json_line = self._fd.readline()

        return json.loads(json_line)

    def __enter__(self):
        self._fd = open(self._filename, "r")

        return self

    def __exit__(self, type, value, traceback):
        if self._fd:
            self._fd.close()

        return True
