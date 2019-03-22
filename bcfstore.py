import numpy as np

class bcf_store_memory():
    def __init__(self, filename):
        self._filename = filename
        print ('Loading BCF file to memory ... '+filename)
        file = open(filename, 'rb')
        size = np.fromstring(file.read(8)[0], dtype=np.uint64)
        file_sizes = np.fromstring(file.read(8*size), dtype=np.uint64)
        self._offsets = np.append(np.uint64(0), np.add.accumulate(file_sizes))
        self._memory = file.read()
        file.close()

    def get(self, i):
        return self._memory[self._offsets[i]:self._offsets[i+1]]

    def size(self):
        return len(self._offsets)-1

class bcf_store_file():
    def __init__(self, filename):
        self._filename = filename
        print ('Opening BCF file ... '+filename)
        self._file = open(filename, 'rb')
        size = np.fromstring(self._file.read(8), dtype=np.uint64)[0]
        size = int(size)
        file_sizes = np.fromstring(self._file.read(8*size), dtype=np.uint64)
        #to reduce the dataset
        #file_sizes = file_sizes[:51000]
        self._offsets = np.append(np.uint64(0), np.add.accumulate(file_sizes))
        #self.orig_size = len(self._offsets)*8
        #self._offsets = self._offsets[:51001]

    def __del__(self):
        self._file.close()

    def get(self, i):
        self._file.seek(len(self._offsets)*8+int(self._offsets[i]))
        #self._file.seek(self.orig_size+int(self._offsets[i]))
        return self._file.read(int(self._offsets[i+1]-self._offsets[i]))

    def size(self):
        return len(self._offsets)-1
