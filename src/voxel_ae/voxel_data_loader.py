import tarfile
import zlib
import numpy as np
import cStringIO as StringIO
import copy


class VoxelLoader():
    # Reference: http://www.cvc.uab.es/people/joans/
    # slides_tensorflow/tensorflow_html/feeding_and_queues.html

    def __init__(self, data_path):
        self.data_path = data_path 
        self.load_data()
        self.num_samples = self.samples.shape[0]
        self.epochs_completed = 0
        self.shuffle()
        self.index_in_epoch = 0
        self.starts_new_epoch = True


    def load_data(self):
        tfile = tarfile.open(self.data_path, 'r|')
        PREFIX = 'data/'
        SUFFIX = '.npy.z'
        self.samples = []
        self.sample_names = []
        for entry in tfile:
            name = entry.name[len(PREFIX):-len(SUFFIX)]
            fileobj = tfile.extractfile(entry)
            buf = zlib.decompress(fileobj.read())
            voxel_grid = np.load(StringIO.StringIO(buf))
            self.samples.append(voxel_grid)
            self.sample_names.append(name)
        tfile.close()
        self.samples = np.asarray(self.samples) 
        self.samples = np.expand_dims(self.samples, -1)


    def shuffle(self):
        randperm = np.random.permutation(self.num_samples)
        self.samples = self.samples[randperm]
 

    def next_batch(self, batch_size, preprocess=True):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_samples:
            # Finished epoch
            self.starts_new_epoch = True
            self.epochs_completed += 1            
            self.shuffle()
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_samples
        else:
            self.starts_new_epoch = False

        end = self.index_in_epoch        
        return self.load_batches(start, end, preprocess)


    def load_batches(self, start, end, preprocess):
        samples = []
        names = []
        for i in xrange(start, end):
            sample = self.samples[i]
            name = self.sample_names[i]
            if preprocess:
                sample = self.preprocess(sample)
            samples.append(sample)
            names.append(name)
            
        return np.asarray(samples), names


    def preprocess(self, sample):
        return 3. * sample - 1.
        #return 2. * sample - 1.
        #return 5. * sample - 1.
        #proc_sample = copy.deepcopy(sample).astype('float')
        #proc_sample[sample == 1] = 6.
        #proc_sample[sample == 0] = -1.
        #return proc_sample

