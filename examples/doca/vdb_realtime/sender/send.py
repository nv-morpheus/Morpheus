#!/usr/bin/python

import glob
import os
import socket
import sys

from scapy.all import *


def main(args):
    os.chdir("dataset")
    for file in glob.glob("*.txt"):
        fp = open(file, 'r')
        while True:
            content = fp.read(1024)
            if not content:
                break
            pkt = IP(src="192.168.2.28", dst="192.168.2.27") / UDP(sport=RandShort(),
                                                                   dport=5001) / Raw(load=content.encode('utf-8'))
            print(pkt)
            send(pkt, iface="enp202s0f0np0")
            #sock.sendto(line.encode('utf-8'), (ip, port))
        fp.close()


main(sys.argv)
