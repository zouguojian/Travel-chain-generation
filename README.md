# Travel-chain-generation
Using traffic flow and travel preference data to generate an individual travel chain on the highway

# Data alignment
To integrate traffic state data with vehicle trajectory data, we performed temporal alignment and data fusion. For each complete trajectory in the trajectory dataset, we used the timestamp at which the vehicle passed the starting gantry to extract historical traffic state data, both speed and flow, for the preceding 12 time slices (equivalent to 1 hour). This process provided dynamic traffic environment characteristics for each trajectory prior to departure, including historical speed distributions across each section and traffic flow conditions at each node. As a result, two aligned traffic state datasets were generated, containing speed and flow information for the 12 time slices preceding each trajectory’s departure.

# Trajectory structure
# 项目根架构：78000F（核心服务平台）
├── **接入层** (2002)
│   ├── 负载均衡：Nginx  
│   └── 安全网关：OpenResty  
├── **核心服务引擎** (780011)
│   ├── 用户事务模块 (78005D)
│   │   ├── 鉴权服务：JWT/OAuth2.0  
│   │   └── 事务管理：Spring Transaction  
│   ├── 数据处理中心 (780013)
│   │   ├── 实时计算引擎 (2008)：Flink流处理  
│   │   ├── 存储集群 (780019)
│   │   │   ├── 分布式存储节点 (78001B)
│   │   │   │   ├── 缓存服务 (78001D)：Redis Cluster  
│   │   │   │   ├── 数据库服务 (101001)：MySQL分片  
│   │   │   │   └── 加密服务 (790064)：TLS证书管理  
│   │   │   └── 运维监控 (101007)：Prometheus+Grafana  
│   │   └── 日志管道 (2007)：ELK Stack  
│   └── 消息通信层 (2009)：RabbitMQ/Kafka  
├── **实验性功能分支** (独立分支节点)
│   └── 灰度发布模块 (78005D_重复)  
│       └── 特性开关：LaunchDarkly集成  
└── **终端控制节点**  
    ├── 管理控制台 (2005)：Vue3+Element Plus  
    └── 客户端SDK (780023)：Java/Python库
