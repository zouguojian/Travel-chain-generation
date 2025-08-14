# Travel-chain-generation
Using traffic flow and travel preference data to generate an individual travel chain on the highway

# Data alignment
To integrate traffic state data with vehicle trajectory data, we performed temporal alignment and data fusion. For each complete trajectory in the trajectory dataset, we used the timestamp at which the vehicle passed the starting gantry to extract historical traffic state data, both speed and flow, for the preceding 12 time slices (equivalent to 1 hour). This process provided dynamic traffic environment characteristics for each trajectory prior to departure, including historical speed distributions across each section and traffic flow conditions at each node. As a result, two aligned traffic state datasets were generated, containing speed and flow information for the 12 time slices preceding each trajectory’s departure.

\section*{树结构}
\begin{verbatim}
78000F (根节点)
├── 2002
├── 780011
│   ├── 78005D
│   ├── 780013
│   │   ├── 2008
│   │   ├── 780019
│   │   │   ├── 78001B
│   │   │   │   ├── 78001D
│   │   │   │   │   ├── 78001F
│   │   │   │   │   │   ├── 780021
│   │   │   │   │   │   │   └── 780023
│   │   │   │   │   │   └── 2005
│   │   │   │   │   └── 790064
│   │   │   │   └── 101001
│   │   │   └── 101007
│   │   └── 2007
│   └── 2009
└── (独立分支节点)
    └── 78005D  # 注：此节点同时存在于根节点和780011下
\end{verbatim}
