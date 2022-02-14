import socket

class Sender:
    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, data: str):
        self.s.sendto(data.encode(), (self.ip_address, self.port))


if __name__ == "__main__":
    from itertools import cycle
    import time
    
    ip_address = '127.0.0.1'
    port = 7010

    labels = ("fur", "ful", "flr", "fll", "bur", "bul", "blr", "bll")
    
    sender = Sender(ip_address, port)
    for l1 in cycle(labels):
        for l2 in labels:
            for i in range(10):
                sender.send(f"{l1},{l2}".upper())
                time.sleep(0.5)
