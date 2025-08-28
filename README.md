# Travel-chain-generation
Using traffic flow and travel preference data to generate an individual travel chain on the highway

# Data alignment
To integrate traffic state data with vehicle trajectory data, we performed temporal alignment and data fusion. For each complete trajectory in the trajectory dataset, we used the timestamp at which the vehicle passed the starting gantry to extract historical traffic state data, both speed and flow, for the preceding 12 time slices (equivalent to 1 hour). This process provided dynamic traffic environment characteristics for each trajectory prior to departure, including historical speed distributions across each section and traffic flow conditions at each node. As a result, two aligned traffic state datasets were generated, containing speed and flow information for the 12 time slices preceding each trajectory’s departure.

# Data preprocess
**Step 1:** Process the flow and speed data, with a temporal granularity of five minutes. The rows correspond to time and traffic state information, while the columns represent attribute names. Please refer to the file *data/states/datapro.py*.

**Step 2:** Load each trajectory record and round down each vehicle’s departure time. For example, “2021-06-01 00:30:04” is rounded down to “2021-06-01 00:30:00,” and “2021-06-01 00:27:46” is rounded down to “2021-06-01 00:25:00.” Please refer to *data/Dataload.py*.

**Step 3:** Match the rounded departure time of the trajectory with the traffic state time to obtain the corresponding historical traffic state values over the previous 12 time steps. This procedure can also be found in *data/Dataload.py*.

# Trajectory structure
<p align="left"> <img src="https://github.com/zouguojian/Travel-chain-generation/blob/main/data/trajectory%20structure.svg" alt="zouguojian" /> </p>

# Relative research
If you find this repository useful in your research, please cite the following paper:
```
@article{zou2024mt,
  title={MT-STNet: A Novel Multi-Task Spatiotemporal Network for Highway Traffic Flow Prediction},
  author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={25},
  number={7},
  pages={8221--8236},
  year={2024},
  publisher={IEEE}
}

@article{zou2023will,
  title={When will we arrive? A novel multi-task spatio-temporal attention network based on individual preference for estimating travel time},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Tu, Meiting and Fan, Jing and Li, Ye},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={10},
  pages={11438--11452},
  year={2023},
  publisher={IEEE}
}

@article{zou2023novel,
  title={A novel spatio-temporal generative inference network for predicting the long-term highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Ma, Changxi and Li, Ye and Wang, Ting},
  journal={Transportation research part C: emerging technologies},
  volume={154},
  pages={104263},
  year={2023},
  publisher={Elsevier}
}

@article{zou2024multi,
  title={Multi-task-based spatiotemporal generative inference network: A novel framework for predicting the highway traffic speed},
  author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Bao, Jingjue and Ma, Changxi and Li, Ye and Fan, Jing},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121548},
  year={2024},
  publisher={Elsevier}
}
```
