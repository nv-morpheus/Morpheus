import argparse
import time
import requests
import socket
import struct
import binascii
import sys

import threading
from multiprocessing import Process

#import payload_gen

def gen_payload():
    custom_fake = payload_gen.PayloadProvider(si_probs)
    return custom_fake.payload()

def send_udp_packet(port, debug_interval, debug):
    
    UDP_IP = '192.168.32.10'
    UDP_PORT = int(port) # 5005

    #MESSAGE = b"Sample UDP Datagram, we should not be grabbing this in Morpheus! "
    MESSAGE = b"UDP -- we should not receive this! " * 100

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
        )

    start = time.time()
    pkt_ctr = 0     
    
    while(True):

        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT)) 

        delta = time.time() - start
        pkt_ctr += 1           

        if debug and (delta > debug_interval):
            print(port, pkt_ctr / delta)
            start = time.time()
            pkt_ctr = 0        

def receive_udp_multicast(port, debug_interval, debug):

    MCAST_GRP = '224.1.1.1' 
    MCAST_PORT = 8004

    sock = socket.socket(
        socket.AF_INET, 
        socket.SOCK_DGRAM, 
        socket.IPPROTO_UDP
    )

    sock.setsockopt(
        socket.SOL_SOCKET, 
        socket.SO_REUSEADDR, 
        1
    )

    sock.bind(('', MCAST_PORT))

    mreq = struct.pack(
        "4sl", 
        socket.inet_aton(MCAST_GRP), 
        socket.INADDR_ANY
    )

    sock.setsockopt(
        socket.IPPROTO_IP, 
        socket.IP_ADD_MEMBERSHIP, 
        mreq
    )

    start = time.time()
    pkt_ctr = 0    

    while(True):

        recv_data = sock.recv(1024)

        delta = time.time() - start
        pkt_ctr += 1        

        if debug and (delta > debug_interval):
            print(port, pkt_ctr / delta, recv_data)
            start = time.time()
            pkt_ctr = 0         

def make_tcp_request(port, debug_interval, debug):

    si_probs = {}
    si_probs['si_address'] = 0
    si_probs['si_bank_acct'] = 0
    si_probs['si_credit_card'] = 0
    si_probs['si_email'] = 0.01
    si_probs['si_govt_id'] = 0
    si_probs['si_name'] = 0
    si_probs['si_password'] = .6
    si_probs['si_phone_num'] = 0
    si_probs['si_secret_keys'] = .6
    si_probs['si_user'] = 0

    custom_fake = payload_gen.PayloadProvider(si_probs)
    url = 'http://192.168.32.10:{}/api/v1/any_test'.format(port)

    start = time.time()
    pkt_ctr = 0

    while (True):

        delta = time.time() - start
        pkt_ctr += 1    

        if port == '5000':
            payload = custom_fake.payload()
            response = requests.post(url, json=payload)
        else:
            payload = 'Welcome to the GTC Accelerated Network TAP Demo '* 100
            response = requests.post(url, data=payload)
            

        if debug and (delta > debug_interval):
            print(port, pkt_ctr / delta)
            start = time.time()
            pkt_ctr = 0            

def generate_jobs(njobs=16, ports=[5000, 5001, 5002, 5003]):

    jobs = {}
    nbins = len(ports)

    for i in range(njobs):
        jobs[i] = i%nbins

    return jobs
        
def _request_worker(port, debug, debug_interval, request_type):


    while(True):

        if request_type == 'tcp':
            make_tcp_request(port, debug_interval, debug)
        elif request_type == 'udp':
            send_udp_packet(port, debug_interval, debug)
        else:
            receive_udp_multicast(port, debug_interval, debug)


def _request_loop_async(
    nthreads, port, request_type, debug=True, debug_interval=2):

    workers = []

    for j in range(nthreads):

        w = threading.Thread(
            target=_request_worker, 
            kwargs=dict(
                port=port,
                debug=debug,
                debug_interval=debug_interval,
                request_type=request_type)
            )

        w.start()
        workers.append(w)

    for w in workers:
        w.join()

def generate_traffic(ports, request_type):

    ports = ports #list(set(ports))
    njobs = len(ports) * 2
    jobs = generate_jobs(njobs=njobs, ports=ports)
    debug = True
    debug_interval = 1
    procs = []

    for j in jobs:

        proc = Process(
            target=_request_loop_async,
            kwargs=dict(
                nthreads=2,
                port=ports[jobs[j]], 
                debug=True, 
                debug_interval=2,
                request_type=request_type
                )
            )

        procs.append(proc)
        proc.start()       

    for proc in procs:
        proc.join()   

def main(args):
    generate_traffic(
        ports=args.ports, request_type=args.type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ports',
        '-p',
        nargs='+',
        help='Provide list of ports.',
        required=True
    )
    parser.add_argument(
        '--type',
        '-t',
        help='Provide type of request', 
        choices=('tcp', 'udp', 'udp-multicast'),
        type=str,
        required=True
    )    

    args = parser.parse_args()
    main(args)
