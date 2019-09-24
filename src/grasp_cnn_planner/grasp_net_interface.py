import abc

class GraspNet(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train():
        return

    @abc.abstractmethod
    def test():
        return

