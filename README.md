# multi-fingered_grasp_planner
This repo has the implementation of three different learning-based multi-fingered grasp planners. 

# Probablistic Multi-fingered Grasp Planner

This repo has several learning-based multi-fingered grasp planners implemented. 
We proposed multiple machine leanring models to predict the probability of grasp success from visual information of the object and grasp configuration. 
We then formulated grasp planning as inferring the grasp configuration which maximizes the probability of grasp success inside the grasp prediction deep networks. 

## Requirement 
ROS Kinetic, Python 2.7, Tensorflow 1.13.1, scikit-learn 0.20.3, PCL 1.7.2, OpenCV. 


## Launch Grasp Planners

Command to launch the RGBD-based grasp planner of citation [1]: 

```roslaunch prob_grasp_planner grasp_cnn_inference.launch```

Command to launch the grasp type planner of citation [2]: 

```roslaunch prob_grasp_planner grasp_type_inference.launch```

Command to launch the voxel-based grasp planner of citation [3]: 

```roslaunch prob_grasp_planner grasp_voxel_inference.launch```

## Grasp Planner Project Pages
[Planning Multi-Fingered Grasps as Probabilistic Inference in a Learned Deep Network](https://robot-learning.cs.utah.edu/project/grasp_inference)

[https://robot-learning.cs.utah.edu/project/grasp_type](https://robot-learning.cs.utah.edu/project/grasp_type)

## Citations 

We list the bibtex citations of this repo. 

    [1] @inproceedings{lu2017grasp,    
    title={{[Planning Multi-Fingered Grasps as Probabilistic Inference in a Learned Deep Network](https://robot-learning.cs.utah.edu/project/grasp_inference)}},    
    author={Lu, Qingkai and Chenna, Kautilya and Sundaralingam, Balakumar and Hermans, Tucker},    
    booktitle={Int'l Symp. on Robotics Research},    
    year={2017}    
    }
    
    [2] @article{lu2019grasp,
    title={{Modeling Grasp Type Improves Learning-Based Grasp Planning}},
    author={Lu, Qingkai and Hermans, Tucker},
    journal={IEEE Robotics and Automation Letters},
    year={2019}
    }
    
    [3] @article{lu2019multifinger,
	title={{Multi-Fingered Grasp Planning via Inference in Deep Neural Networks}},
	author={Lu, Qingkai, and Van der Merwe, Mark,  and Sundaralingam, Balakumar and Hermans, Tucker},
	journal={{IEEE} Robotics \& Automation Magazine (Under Review)},
	year={2019}
    }

