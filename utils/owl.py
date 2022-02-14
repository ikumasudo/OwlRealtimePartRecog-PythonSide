import keyboard
import numpy as np
import serial
import threading
import time

SCALE = 32768
MAX_MEM = 1000

class Owl:
    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.stop_flg = False
        self.memory = []

    def open(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            return True
        except serial.serialutil.SerialException:
            return False


    def get_data(self, n: int = -1) -> np.ndarray:
        if n < 0:
            data = np.array(self.memory)
        else:
            data = np.array(self.memory[-n:])
        return data


    def _receive(self):
        while not self.stop_flg:
            if self.ser.in_waiting:
                try:
                    recv = self.ser.readline().decode().strip()
                    vals = [float(i)/SCALE for i in recv.split(",")]
                    
                    self.memory.append(vals)
                    if len(self.memory) > MAX_MEM:
                        self.memory.pop(0)
                except Exception as e:
                    continue
        self.ser.close()

    
    def start(self):
        self.thread = threading.Thread(target=self._receive)
        self.thread.start()


    def stop(self):
        self.stop_flg = True
        self.thread.join()


if __name__ == "__main__":
    import time
    import matplotlib.pyplot
    
    owl = Owl("COM7", 115200)
    owl.start()
    time.sleep(10)
    print(owl.get_data().shape)
    owl.stop()
