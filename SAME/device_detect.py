from torch import backends,device,cuda

class DeviceDetermination():
    def __init__(self):
        pass

    def device_detect(self):
        if backends.mps.is_available():
            return "mps"
        if cuda.is_available():
            return "cuda"
        else:
            return "cpu"
        
    def __call__(self):
        return self.device_detect()
         
