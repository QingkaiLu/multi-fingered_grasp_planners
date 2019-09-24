import numpy as np

class SetConfigLimits: 

    def __init__(self):
        self.palm_loc_dof_dim = 3
        self.palm_dof_dim = 6
        self.finger_joints_dof_dim = 8
        self.config_dim = self.palm_dof_dim + self.finger_joints_dof_dim 
        self.setup_config_limits()
        self.setup_isrr_config_limits()


    def setup_joint_angle_limits(self):
        '''
        Initializes a number of constants determing the joint limits for allegro
        '''
        self.index_joint_0_lower = -0.59
        self.index_joint_0_upper = 0.57
        self.middle_joint_0_lower = -0.59
        self.middle_joint_0_upper = 0.57
        self.ring_joint_0_lower = -0.59
        self.ring_joint_0_upper = 0.57
        
        self.index_joint_1_lower = -0.296
        self.index_joint_1_upper = 0.71
        self.middle_joint_1_lower = -0.296
        self.middle_joint_1_upper = 0.71
        self.ring_joint_1_lower = -0.296
        self.ring_joint_1_upper = 0.71
        
        self.thumb_joint_0_lower = 0.363
        self.thumb_joint_0_upper = 1.55
        self.thumb_joint_1_lower = -0.205
        self.thumb_joint_1_upper = 1.263


    def setup_config_limits(self):
        '''
        Set up the limits for grasp preshape configurations.
        '''
        self.preshape_config_lower_limit = np.zeros(self.config_dim)
        self.preshape_config_upper_limit = np.zeros(self.config_dim)

        # self.preshape_config_lower_limit[:self.palm_dof_dim] = \
        #         np.array([-1., -1., -2., -np.pi, -np.pi, -np.pi])
        # self.preshape_config_upper_limit[:self.palm_dof_dim] = \
        #         np.array([1., 1., 0.5, np.pi, np.pi, np.pi])

        self.setup_palm_pose_limits(self.preshape_config_lower_limit, 
                                    self.preshape_config_upper_limit)

        self.setup_joint_angle_limits()
        self.preshape_config_lower_limit[self.palm_dof_dim] = self.index_joint_0_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 1] = self.index_joint_1_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 2] = self.middle_joint_0_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 3] = self.middle_joint_1_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 4] = self.ring_joint_0_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 5] = self.ring_joint_1_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 6] = self.thumb_joint_0_lower
        self.preshape_config_lower_limit[self.palm_dof_dim + 7] = self.thumb_joint_1_lower

        self.preshape_config_upper_limit[self.palm_dof_dim] = self.index_joint_0_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 1] = self.index_joint_1_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 2] = self.middle_joint_0_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 3] = self.middle_joint_1_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 4] = self.ring_joint_0_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 5] = self.ring_joint_1_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 6] = self.thumb_joint_0_upper
        self.preshape_config_upper_limit[self.palm_dof_dim + 7] = self.thumb_joint_1_upper


    def setup_palm_pose_limits(self, config_lower_limit, config_upper_limit):
        # pos_range = 0.5
        # ort_range = 0.5 * np.pi
        pos_range = float('inf')
        ort_range = np.pi #float('inf')
        config_lower_limit[:self.palm_dof_dim] = \
                        -np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])
        config_upper_limit[:self.palm_dof_dim] = \
                        np.array([pos_range, pos_range, pos_range, ort_range, ort_range, ort_range])


    def setup_isrr_config_limits(self):
        self.isrr_config_lower_limit = np.zeros(self.config_dim)
        self.isrr_config_upper_limit = np.zeros(self.config_dim)

        self.setup_palm_pose_limits(self.isrr_config_lower_limit, 
                                    self.isrr_config_upper_limit)

        #Set up joint limits for our isrr paper
        index_joint_0_middle = (self.index_joint_0_lower + self.index_joint_0_upper) * 0.5
        middle_joint_0_middle = (self.middle_joint_0_lower + self.middle_joint_0_upper) * 0.5
        ring_joint_0_middle = (self.ring_joint_0_lower + self.ring_joint_0_upper) * 0.5
        index_joint_1_middle = (self.index_joint_1_lower + self.index_joint_1_upper) * 0.5
        middle_joint_1_middle = (self.middle_joint_1_lower + self.middle_joint_1_upper) * 0.5
        ring_joint_1_middle = (self.ring_joint_1_lower + self.ring_joint_1_upper) * 0.5
        thumb_joint_0_middle = (self.thumb_joint_0_lower + self.thumb_joint_0_upper) * 0.5
        thumb_joint_1_middle = (self.thumb_joint_1_lower + self.thumb_joint_1_upper) * 0.5

        index_joint_0_range = self.index_joint_0_upper - self.index_joint_0_lower
        middle_joint_0_range = self.middle_joint_0_upper - self.middle_joint_0_lower
        ring_joint_0_range = self.ring_joint_0_upper - self.ring_joint_0_lower
        index_joint_1_range = self.index_joint_1_upper - self.index_joint_1_lower
        middle_joint_1_range = self.middle_joint_1_upper - self.middle_joint_1_lower
        ring_joint_1_range = self.ring_joint_1_upper - self.ring_joint_1_lower
        thumb_joint_0_range = self.thumb_joint_0_upper - self.thumb_joint_0_lower
        thumb_joint_1_range = self.thumb_joint_1_upper - self.thumb_joint_1_lower

        first_joint_lower_limit = 0.5
        first_joint_upper_limit = 0.5
        second_joint_lower_limit = 0.5
        second_joint_upper_limit = 0.

        thumb_1st_joint_lower_limit = 0.
        thumb_1st_joint_upper_limit = 1.0
        thumb_2nd_joint_lower_limit = 0.5
        thumb_2nd_joint_upper_limit = 0.5

        self.isrr_config_lower_limit[self.palm_dof_dim] = \
                index_joint_0_middle - first_joint_lower_limit * index_joint_0_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 1] = \
                index_joint_1_middle - second_joint_lower_limit * index_joint_1_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 2] = \
                middle_joint_0_middle - first_joint_lower_limit * middle_joint_0_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 3] = \
                middle_joint_1_middle - second_joint_lower_limit * middle_joint_1_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 4] = \
                ring_joint_0_middle - first_joint_lower_limit * ring_joint_0_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 5] = \
                ring_joint_1_middle - second_joint_lower_limit * ring_joint_1_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 6] = \
                thumb_joint_0_middle - thumb_1st_joint_lower_limit * thumb_joint_0_range
        self.isrr_config_lower_limit[self.palm_dof_dim + 7] = \
                thumb_joint_1_middle - thumb_2nd_joint_lower_limit * thumb_joint_1_range
                
        self.isrr_config_upper_limit[self.palm_dof_dim] = \
                index_joint_0_middle + first_joint_upper_limit * index_joint_0_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 1] = \
                index_joint_1_middle + second_joint_upper_limit * index_joint_1_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 2] = \
                middle_joint_0_middle + first_joint_upper_limit * middle_joint_0_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 3] = \
                middle_joint_1_middle + second_joint_upper_limit * middle_joint_1_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 4] = \
                ring_joint_0_middle + first_joint_upper_limit * ring_joint_0_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 5] = \
                ring_joint_1_middle + second_joint_upper_limit * ring_joint_1_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 6] = \
                thumb_joint_0_middle + thumb_1st_joint_upper_limit * thumb_joint_0_range
        self.isrr_config_upper_limit[self.palm_dof_dim + 7] = \
                thumb_joint_1_middle + thumb_2nd_joint_upper_limit * thumb_joint_1_range


