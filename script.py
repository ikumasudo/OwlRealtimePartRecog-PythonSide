import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import keyboard
import time

from utils.preprocess import Preprocess
from utils.classifier import Classifier
from utils.owl import Owl
from utils.udp import Sender


def main():
    owl = Owl("COM7", 115200)
    while True:
        if keyboard.is_pressed("esc"):
            return
        
        if owl.open():
            print(f"Connected to {owl.port}")
            break
        else:
            print(f"{owl.port} is not available. ")
            time.sleep(0.5)
    
    ip_address = '127.0.0.1'
    port = 7010
    sender = Sender(ip_address, port)
    
    print("Starting owl...")
    owl.start()
    
    prep = Preprocess(".\data\OwlNotebook20-FeatureImportance.csv", 300)
    clf = Classifier("./data/OwlNotebook20-LR-300.pickle")
    
    while True:
        if keyboard.is_pressed("esc"):
            break
        
        data = owl.get_data(100).transpose()
        if len(data) == 0:
            continue
        # plt.plot(data.transpose())
        # plt.show()

        
        data = prep(data)

        if data is not None:
            pred = clf.predict(data)
            print(pred[-1])
            sender.send(pred[-1].upper())
        else:
            print("No peaks detected")
            
        time.sleep(0.05)
    owl.stop()

if __name__ == "__main__":
    main()