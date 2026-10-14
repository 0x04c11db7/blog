---
title: TC Pedit 实现无状态 NAT
date: 2024-09-20 10:36:51
tags: Network
summary: 为了让 TCP 回程数据包走不同的路由，尝试用 Linux TC 的 Pedit 修改源和目标 IP。记录如何在 egress 方向匹配数据包、改写地址并重新计算校验和，实现不依赖 Conntrack 的无状态 NAT。
---
　　最近有一个需求，修改 TCP 回程数据包的 SRC 和 DST IP 使与去程走不一样的路由。第一反应是使用 iptables 做 SNAT 和 DNAT，但是实测发现 iptables 是基于 Conntrack 实现的，对回程数据包并不生效。因为曾经用 XDP 实现过 LB，准备再次使用 XDP 写个小程序实现。

　　随后突然想起 ebpf 的 XDP Hook 是对 ingress 生效的，如果想做 egress 则得使用 TC Hook。既然如此， Linux 的 TC 工具是不是已经原生支持了我的小需求呢，之前对 TC 工具的使用局限在 netem 和 tbf 模块，也就是弱网模拟和流控。查阅文档发现 TC 的 Pedit 模块可以很轻松的提供我需要的无状态（Stateless）NAT 能力。

<!-- more --> 

　　在我的场景下需要将 wg0 发出的数据包的 SRC 和 DST 修改成特定的 IP，这里就以匹配 SRC 3.3.3.3 和 DST 4.4.4.4 的数据包并将其修改为 SRC 1.1.1.1 和 DST 2.2.2.2 为例了。
## 创建队列
　　首先就是创建一个 clsact 队列，这个队列有 ingress 和 egress 两个钩子。

```bash
$ tc qdisc add dev wg0 clsact
```

## 修改数据包
　　随后的命令很直观，指定网卡 wg0 的 egress 方向。flower（flow based traffic control filter）根据 src_ip dst_ip 匹配 ip 数据包，首先通过 pedit action 修改 ip，再使用 csum action 为数据包重新计算 Checksum。

```bash
$ tc filter add dev wg0 egress protocol ip \
    flower \
    src_ip 3.3.3.3 \
    dst_ip 4.4.4.4 \
    action pedit ex \
        munge ip src set 1.1.1.1 \
        munge ip dst set 2.2.2.2 \
    pipe \
    action csum ip tcp udp
```
