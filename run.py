#!/usr/bin/env python3

from roop import core
import time 

if __name__ == '__main__':   
    t1 = time.time()
    core.run()
    print(f"Total processing time: {time.time() - t1:.2f} seconds")
      

#import onnxruntime as ort
#print(ort.get_device())  # Should print "GPU"
