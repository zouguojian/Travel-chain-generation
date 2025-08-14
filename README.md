# Travel-chain-generation
Using traffic flow and travel preference data to generate an individual travel chain on the highway

# Data alignment
To integrate traffic state data with vehicle trajectory data, we performed temporal alignment and data fusion. For each complete trajectory in the trajectory dataset, we used the timestamp at which the vehicle passed the starting gantry to extract historical traffic state data, both speed and flow, for the preceding 12 time slices (equivalent to 1 hour). This process provided dynamic traffic environment characteristics for each trajectory prior to departure, including historical speed distributions across each section and traffic flow conditions at each node. As a result, two aligned traffic state datasets were generated, containing speed and flow information for the 12 time slices preceding each trajectory’s departure.

# Trajectory structure
<p align="left"> <img src="https://github.com/zouguojian/Travel-chain-generation/blob/main/data/trajectory%20structure.svg" alt="zouguojian" /> </p>
