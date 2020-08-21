
# custom data structure to hold the state of a Canny edge filter
class EdgeFilter:

    def __init__(self, kernelSize=None, erodeIter=None, dilateIter=None, canny1=None, 
                    canny2=None):
        self.kernelSize = kernelSize
        self.erodeIter = erodeIter
        self.dilateIter = dilateIter
        self.canny1 = canny1
        self.canny2 = canny2
