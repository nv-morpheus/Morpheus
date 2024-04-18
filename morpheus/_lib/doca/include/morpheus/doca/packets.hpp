/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define ETHER_ADDR_LEN  6 /**< Length of Ethernet address. */
#define IP_ADDR_STRING_LEN  16

#define BYTE_SWAP16(v) \
    ((((uint16_t)(v) & UINT16_C(0x00ff)) << 8) | (((uint16_t)(v) & UINT16_C(0xff00)) >> 8))

#define TCP_PROTOCOL_ID 0x6
#define UDP_PROTOCOL_ID 0x11
#define TIMEOUT_NS 500000 //500us
#define RTE_IPV4_HDR_IHL_MASK	(0x0f)
#define RTE_IPV4_IHL_MULTIPLIER	(4)

enum tcp_flags {
    TCP_FLAG_FIN = (1 << 0),
    /* set tcp packet with Fin flag */
    TCP_FLAG_SYN = (1 << 1),
    /* set tcp packet with Syn flag */
    TCP_FLAG_RST = (1 << 2),
    /* set tcp packet with Rst flag */
    TCP_FLAG_PSH = (1 << 3),
    /* set tcp packet with Psh flag */
    TCP_FLAG_ACK = (1 << 4),
    /* set tcp packet with Ack flag */
    TCP_FLAG_URG = (1 << 5),
    /* set tcp packet with Urg flag */
    TCP_FLAG_ECE = (1 << 6),
    /* set tcp packet with ECE flag */
    TCP_FLAG_CWR = (1 << 7),
    /* set tcp packet with CQE flag */
};

struct ether_hdr {
    uint8_t d_addr_bytes[ETHER_ADDR_LEN];	/* Destination addr bytes in tx order */
    uint8_t s_addr_bytes[ETHER_ADDR_LEN];	/* Source addr bytes in tx order */
    uint16_t ether_type;			/* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr {
    uint8_t version_ihl;		/* version and header length */
    uint8_t  type_of_service;	/* type of service */
    uint16_t total_length;		/* length of packet */
    uint16_t packet_id;		/* packet ID */
    uint16_t fragment_offset;	/* fragmentation offset */
    uint8_t  time_to_live;		/* time to live */
    uint8_t  next_proto_id;		/* protocol ID */
    uint16_t hdr_checksum;		/* header checksum */
    uint32_t src_addr;		/* source address */
    uint32_t dst_addr;		/* destination address */
} __attribute__((__packed__));

struct eth_ip {
    struct ether_hdr l2_hdr;	/* Ethernet header */
    struct ipv4_hdr l3_hdr;		/* IP header */
} __attribute__((__packed__));

struct tcp_hdr {
    uint16_t src_port;	/* TCP source port */
    uint16_t dst_port;	/* TCP destination port */
    uint32_t sent_seq;	/* TX data sequence number */
    uint32_t recv_ack;	/* RX data acknowledgment sequence number */
    uint8_t dt_off;		/* Data offset */
    uint8_t tcp_flags;	/* TCP flags */
    uint16_t rx_win;	/* RX flow control window */
    uint16_t cksum;		/* TCP checksum */
    uint16_t tcp_urp;	/* TCP urgent pointer, if any */
} __attribute__((__packed__));

struct eth_ip_tcp_hdr {
    struct ether_hdr l2_hdr;	/* Ethernet header */
    struct ipv4_hdr l3_hdr;		/* IP header */
    struct tcp_hdr l4_hdr;		/* TCP header */
} __attribute__((__packed__));

struct udp_hdr {
    uint16_t src_port;	/* UDP source port */
    uint16_t dst_port;	/* UDP destination port */
    uint16_t dgram_len;	/* UDP datagram length */
    uint16_t dgram_cksum;	/* UDP datagram checksum */
} __attribute__((__packed__));

struct eth_ip_udp_hdr {
    struct ether_hdr l2_hdr;	/* Ethernet header */
    struct ipv4_hdr l3_hdr;		/* IP header */
    struct udp_hdr l4_hdr;		/* UDP header */
} __attribute__((__packed__));

#define TCP_HDR_SIZE sizeof(struct eth_ip_tcp_hdr)
#define UDP_HDR_SIZE sizeof(struct eth_ip_udp_hdr)

__device__ __inline__ int
raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr **hdr, uint8_t **packet_data)
{
    (*hdr) = (struct eth_ip_tcp_hdr *) buf_addr;
    (*packet_data) = (uint8_t *) (buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

    return 0;
}

__device__ __inline__ int
raw_to_udp(const uintptr_t buf_addr, struct eth_ip_udp_hdr **hdr, uint8_t **packet_data)
{
    (*hdr) = (struct eth_ip_udp_hdr *) buf_addr;
    (*packet_data) = (uint8_t *) (buf_addr + sizeof(struct eth_ip_udp_hdr));

    return 0;
}

__device__ __forceinline__ uint8_t
gpu_ipv4_hdr_len(const struct ipv4_hdr& packet_l3)
{
    return (uint8_t)((packet_l3.version_ihl & RTE_IPV4_HDR_IHL_MASK) * RTE_IPV4_IHL_MULTIPLIER);
};

__device__ __forceinline__ uint32_t
get_packet_size(ipv4_hdr& packet_l3)
{
    return static_cast<int32_t>(BYTE_SWAP16(packet_l3.total_length));
}

__device__ __forceinline__ int32_t
get_payload_tcp_size(ipv4_hdr& packet_l3, tcp_hdr& packet_l4)
{
    auto packet_size       = get_packet_size(packet_l3);
    auto ip_header_length  = gpu_ipv4_hdr_len(packet_l3);
    auto tcp_header_length = static_cast<int32_t>(packet_l4.dt_off >> 4) * sizeof(int32_t);
    auto payload_size      = packet_size - ip_header_length - tcp_header_length;

    return payload_size;
}

__device__ __forceinline__ int32_t
get_payload_udp_size(ipv4_hdr& packet_l3, udp_hdr& packet_l4)
{
    auto packet_size       = get_packet_size(packet_l3);
    auto ip_header_length  = gpu_ipv4_hdr_len(packet_l3);
    auto payload_size      = packet_size - ip_header_length - sizeof(struct udp_hdr);

    return payload_size;
}

__device__ __forceinline__ char to_hex_16(uint8_t value)
{
    return "0123456789ABCDEF"[value];
}

__device__ __forceinline__ int64_t mac_bytes_to_int64(uint8_t* mac)
{
    return static_cast<uint64_t>(mac[0]) << 40
        | static_cast<uint64_t>(mac[1]) << 32
        | static_cast<uint32_t>(mac[2]) << 24
        | static_cast<uint32_t>(mac[3]) << 16
        | static_cast<uint32_t>(mac[4]) << 8
        | static_cast<uint32_t>(mac[5]);
}

__device__ __forceinline__ int64_t mac_int64_to_chars(int64_t mac, char* out)
{
    uint8_t mac_0 = (mac >> 40) & (0xFF);
    out[0]  = to_hex_16(mac_0 / 16);
    out[1]  = to_hex_16(mac_0 % 16);
    out[2]  = ':';

    uint8_t mac_1 = (mac >> 32) & (0xFF);
    out[3]  = to_hex_16(mac_1 / 16);
    out[4]  = to_hex_16(mac_1 % 16);
    out[5]  = ':';

    uint8_t mac_2 = (mac >> 24) & (0xFF);
    out[6]  = to_hex_16(mac_2 / 16);
    out[7]  = to_hex_16(mac_2 % 16);
    out[8]  = ':';

    uint8_t mac_3 = (mac >> 16) & (0xFF);
    out[9]  = to_hex_16(mac_3 / 16);
    out[10] = to_hex_16(mac_3 % 16);
    out[11] = ':';

    uint8_t mac_4 = (mac >> 8) & (0xFF);
    out[12] = to_hex_16(mac_4 / 16);
    out[13] = to_hex_16(mac_4 % 16);
    out[14] = ':';

    uint8_t mac_5 = (mac >> 0) & (0xFF);
    out[15] = to_hex_16(mac_5 / 16);
    out[16] = to_hex_16(mac_5 % 16);
}

__device__ __forceinline__ uint32_t ip_to_int32(uint32_t address)
{
    return (address & 0x000000ff) << 24
        | (address & 0x0000ff00) << 8
        | (address & 0x00ff0000) >> 8
        | (address & 0xff000000) >> 24;
}

__device__ __forceinline__ int num_to_string(uint32_t value, char *sp)
{
    char tmp[16];// be careful with the length of the buffer
    char *tp = tmp;
    int i;
    int radix = 10;

    while (value || tp == tmp)
    {
        i = value % radix;
        value /= radix;
        if (i < 10)
          *tp++ = i+'0';
        else
          *tp++ = i + 'a' - 10;
    }

    int len = tp - tmp;
    while (tp > tmp)
        *sp++ = *--tp;

    return len;
}


__device__ __forceinline__ int ip_to_string(uint32_t ip_int, uint8_t *ip_str)
{
    int i;
    int pos = 0;
    int post = 0;
    int radix = 10;
    uint8_t tmp[3];

    // uint32_t ip = ip_to_int32(ip_int);
    // Assuming network order
    uint8_t ip0 = (uint8_t)(ip_int & 0x000000ff);
    uint8_t ip1 = (uint8_t)((ip_int & 0x0000ff00) >> 8);
    uint8_t ip2 = (uint8_t)((ip_int & 0x00ff0000) >> 16);
    uint8_t ip3 = (uint8_t)((ip_int & 0xff000000) >> 24);

    post = 0;
    while (ip0) {
        i = ip0 % radix;
        ip0 /= radix;
        if (i < 10)
          tmp[post++] = i+'0';
        else
          tmp[post++] = i+'a'-10;
    }
    --post;
    while (post >= 0)
        ip_str[pos++] = tmp[post--];
    ip_str[pos++] = '.';
    
    post = 0;
    while (ip1) {
        i = ip1 % radix;
        ip1 /= radix;
        if (i < 10)
          tmp[post++] = i+'0';
        else
          tmp[post++] = i+'a'-10;
    }
    --post;
    while (post >= 0)
        ip_str[pos++] = tmp[post--];
    ip_str[pos++] = '.';
    
    post = 0;
    while (ip2) {
        i = ip2 % radix;
        ip2 /= radix;
        if (i < 10)
          tmp[post++] = i+'0';
        else
          tmp[post++] = i+'a'-10;
    }
    --post;
    while (post >= 0)
        ip_str[pos++] = tmp[post--];
    ip_str[pos++] = '.';
    
    post = 0;
    while (ip3) {
        i = ip3 % radix;
        ip3 /= radix;
        if (i < 10)
          tmp[post++] = i+'0';
        else
          tmp[post++] = i+'a'-10;
    }
    --post;
    while (post >= 0)
        ip_str[pos++] = tmp[post--];

    // printf("ip_str %c%c%c%c %c%c%c%c %c%c%c%c %c%c%c Final pos %d\n",
    //         ip_str[0],ip_str[1],ip_str[2],ip_str[3],
    //         ip_str[4],ip_str[5],ip_str[6],ip_str[7],
    //         ip_str[8],ip_str[9],ip_str[10],ip_str[11],
    //         ip_str[12],ip_str[13],ip_str[14],
    //         pos);

    return pos;
}