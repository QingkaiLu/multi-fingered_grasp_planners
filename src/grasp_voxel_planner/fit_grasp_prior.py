from sklearn import mixture
import pickle
from grasp_data_loader import GraspDataLoader 
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('prob_grasp_planner')
import time

def fit_gmm_prior(grasp_configs, prior_model_path):
    num_components = 2 #3
    #g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
    #        random_state=0, init_params='random', n_init=5)
    g = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
                random_state=0, init_params='kmeans', n_init=5)

    g.fit(grasp_configs)
    #pred_prob = g.predict_proba(grasp_configs)
    #print pred_prob
    #print g.score_samples(grasp_configs)
    print 'weights:', g.weights_
    print 'means:', g.means_
    pickle.dump(g, open(prior_model_path, 'wb'))
    #update_prior_poses_client(g.means_)


if __name__ == '__main__':
    # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
    #                 'merged_grasp_data_6_6_and_6_8.h5'
    # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
    #     'merged_grasp_data_6_6_and_6_8_and_6_10_and_6_11_and_6_13.h5'
    # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
    #                 'merged_grasp_data_10_sets.h5'
    # train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
    #                 'merged_suc_grasp_10_sets.h5'
    train_data_path = '/mnt/tars_data/gazebo_al_grasps/train/' + \
                    'merged_failure_grasp_5_sets.h5'

    grasp_loader = GraspDataLoader(train_data_path)
    # prior_model_path = pkg_path + '/models/grasp_al_prior/gmm_2_sets'
    # prior_model_path = pkg_path + '/models/grasp_al_prior/prior_2_sets'
    # prior_model_path = pkg_path + '/models/grasp_al_prior/gmm_10_sets'
    #prior_model_path = pkg_path + '/models/grasp_al_prior/suc_gmm_10_sets'
    
    prior_model_path = pkg_path + '/models/grasp_al_prior/failure_gmm_5_sets'

    grasp_configs = grasp_loader.load_grasp_configs()
    fit_gmm_prior(grasp_configs, prior_model_path)

    # num_components = 2
    # g1 = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
    #             random_state=0)

    # start_time = time.time()
    # g1.fit(grasp_configs[:-20])
    # elapsed_time = time.time() - start_time
    # print 'g1 fit: ', elapsed_time

    # g2 = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
    #             random_state=0)
    # g2.means_init = g1.means_
    # g2.weights_init = g1.weights_

    # start_time = time.time()
    # g2.fit(grasp_configs)
    # elapsed_time = time.time() - start_time
    # print 'g2 fit: ', elapsed_time

    # g3 = mixture.GaussianMixture(n_components=num_components, covariance_type='full', 
    #             random_state=0, init_params='kmeans', n_init=5)

    # start_time = time.time()
    # g3.fit(grasp_configs)
    # elapsed_time = time.time() - start_time
    # print 'g3 fit: ', elapsed_time



