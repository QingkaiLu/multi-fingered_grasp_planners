from grasp_pgm_inference import GraspPgmInfer 
import tf
import numpy as np
import h5py

def convert_config_to_array(hand_js, palm_pose):
    palm_quaternion = palm_pose[3:]
    palm_euler = tf.transformations.euler_from_quaternion(palm_quaternion)

    preshape_config = [palm_pose[0], palm_pose[1], palm_pose[2],
            palm_euler[0], palm_euler[1], palm_euler[2],
            hand_js[0], hand_js[1],
            hand_js[4], hand_js[5],
            hand_js[8], hand_js[9],
            hand_js[12], hand_js[13]]

    return np.array(preshape_config)


def get_suc_prob():
    
    objects_list = ['lego', 'mustard', 'pitcher', 'pringle', 'scrub']
    #objects_list = ['p']
    #objects_list = ['test']
    grasp_control_types_list = ['prec_prec', 'power_power', 'all_prec', 'all_power']
    grasp_data_pre_path = '/dataspace/data_kai/corl_exp/data/'

    heu_log_priors = []
    heu_clf_log_probs = []
    heu_suc_log_probs = []
    log_priors = []
    clf_log_probs = []
    suc_log_probs = []
    for obj in objects_list:
        print '---------------------------------------------------------'
        for i in xrange(5):
            print '#############################'
            obj_trial_id = 0
            if obj == 'lego' and i == 4:
                obj_trial_id = 1
            if obj == 'pitcher' and i == 1:
                obj_trial_id = 1
            if obj == 'p' and i == 2:
                obj_trial_id = 1
            grasp_data_obj_path = grasp_data_pre_path + obj + '/multi_finger_exp_data_' + obj + '_' + str(i + 1) + \
                    '/grasp_data/object_' + str(obj_trial_id) + '_'

            #Read the heursitic intialization grasp.
            grasp_file_path = grasp_data_obj_path + obj + '_grasp_4_heuristic_prec.h5'
            grasp_data_file = h5py.File(grasp_file_path, 'r')
            heu_obj_grasp_key = 'object_' + str(obj_trial_id) + '_grasp_4'
            heu_hand_js = grasp_data_file[heu_obj_grasp_key + '_preshape_joint_state_position'][()]
            heu_palm_pose = grasp_data_file[heu_obj_grasp_key + '_preshape_palm_pose'][()]
            heu_grasp_config = convert_config_to_array(heu_hand_js, heu_palm_pose)
            grasp_data_file.close()

            for j, gc_type in enumerate(grasp_control_types_list):
                print obj, 'obj_pose_' + str(i + 1), gc_type 
                grasp_file_path = grasp_data_obj_path + obj + '_grasp_' + str(j) + '_' + gc_type + '.h5'
                grasp_data_file = h5py.File(grasp_file_path, 'r')
                obj_grasp_key = 'object_' + str(obj_trial_id) + '_grasp_' + str(j)
                grasp_type = grasp_data_file[obj_grasp_key + '_grasp_type'][()] 
                print 'grasp_type:', grasp_type
                if grasp_type == 'all':
                    grasp_pgm_inf = GraspPgmInfer(pgm_grasp_type=False)
                elif grasp_type == 'prec' or grasp_type == 'power':
                    grasp_pgm_inf = GraspPgmInfer(pgm_grasp_type=True) 
                else:
                    rospy.logerr('Wrong grasp type!')
                
                #Intialization
                heu_log_prior = grasp_pgm_inf.compute_grasp_log_prior(grasp_type, heu_grasp_config)
                print 'heu_log_prior:', heu_log_prior
                heu_suc_log_prob = grasp_data_file[obj_grasp_key + '_inits_suc_prob_list'][()][0]
                print 'heu_suc_log_prob:', heu_suc_log_prob
                heu_clf_log_prob = heu_suc_log_prob - heu_log_prior 
                print 'heu_clf_log_prob:', heu_clf_log_prob

                #Final inference
                hand_js = grasp_data_file[obj_grasp_key + '_preshape_joint_state_position'][()]
                palm_pose = grasp_data_file[obj_grasp_key + '_preshape_palm_pose'][()]
                grasp_config = convert_config_to_array(hand_js, palm_pose)
                log_prior = grasp_pgm_inf.compute_grasp_log_prior(grasp_type, grasp_config)
                print 'grasp_config:', grasp_config
                print 'log_prior:', log_prior
                suc_log_prob = grasp_data_file[obj_grasp_key + '_inf_suc_prob'][()]
                print 'suc_log_prob:', suc_log_prob
                clf_log_prob = suc_log_prob - log_prior 
                print 'clf_log_prob:', clf_log_prob
                grasp_data_file.close()
                print '***********'

                if gc_type == 'prec_prec':
                    heu_log_priors.append(heu_log_prior)
                    heu_clf_log_probs.append(heu_clf_log_prob)
                    heu_suc_log_probs.append(heu_suc_log_prob)
                    log_priors.append(log_prior)
                    clf_log_probs.append(clf_log_prob)
                    suc_log_probs.append(suc_log_prob)

    print 'heuristic: log prior, log clf, log suc prob'
    print np.mean(heu_log_priors), np.mean(heu_clf_log_probs), np.mean(heu_suc_log_probs)
    print 'inference: log prior, log clf, log suc prob'
    print np.mean(log_priors), np.mean(clf_log_probs), np.mean(suc_log_probs)

def test_suc_prob():

    #/dataspace/data_kai/corl_exp/data/lego/multi_finger_exp_data_lego_1/grasp_data/object_0_lego_grasp_0_prec_prec.h5

    #if req.grasp_type == 'all':
    #    grasp_pgm_inf = self.grasp_pgm_inf_no_type
    #elif req.grasp_type == 'prec' or req.grasp_type == 'power':
    #    grasp_pgm_inf = self.grasp_pgm_inf_type
    #else:
    #    rospy.logerr('Wrong grasp type for inference!')

    #grasp_pgm_inf = GraspPgmInfer(pgm_grasp_type=True) 

    #grasp_file_path = '/dataspace/data_kai/corl_exp/data/lego/multi_finger_exp_data_lego_1/' + \
    #                    'grasp_data/object_0_lego_grasp_0_prec_prec.h5'
    #grasp_file_path = '/dataspace/data_kai/corl_exp/data/pringle/multi_finger_exp_data_pringle_1/grasp_data/object_0_pringle_grasp_0_prec_prec.h5'
    grasp_file_path = '/dataspace/data_kai/multi_finger_exp_data/grasp_data/object_0_pringle_grasp_0_prec_prec.h5'
    grasp_data_file = h5py.File(grasp_file_path, 'r')
    grasp_pgm_inf = GraspPgmInfer(pgm_grasp_type=True) 
    
    grasp_type = grasp_data_file['object_0_grasp_0_grasp_type'][()] 
    #latent_voxel = 
    #grasp_config = 
    hand_js = grasp_data_file['object_0_grasp_0_preshape_joint_state_position'][()]
    palm_pose = grasp_data_file['object_0_grasp_0_preshape_palm_pose'][()]
    grasp_config = convert_config_to_array(hand_js, palm_pose)
    #clf_log_prob = self.compute_clf_log_suc_prob(grasp_type, latent_voxel, grasp_config)
    log_prior = grasp_pgm_inf.compute_grasp_log_prior(grasp_type, grasp_config)
    #print 'clf_log_prob:', clf_log_prob
    print 'log_prior:', log_prior
    #obj_prob = clf_log_prob + log_prior 
    suc_log_prob = grasp_data_file['object_0_grasp_0_inf_suc_prob'][()]
    print 'suc_log_prob:', suc_log_prob
    clf_log_prob = suc_log_prob - log_prior 
    print 'clf_log_prob:', clf_log_prob
    grasp_data_file.close()

    #grasp_file_path = '/dataspace/data_kai/corl_exp/data/lego/multi_finger_exp_data_lego_1/' + \
    #                    'grasp_data/object_0_lego_grasp_1_power_power.h5'
    #grasp_data_file = h5py.File(grasp_file_path, 'r')
    #grasp_type = grasp_data_file['object_0_grasp_1_grasp_type'][()] 
    ##latent_voxel = 
    ##grasp_config = 
    #hand_js = grasp_data_file['object_0_grasp_1_preshape_joint_state_position'][()]
    #palm_pose = grasp_data_file['object_0_grasp_1_preshape_palm_pose'][()]
    #grasp_config = convert_config_to_array(hand_js, palm_pose)
    ##clf_log_prob = self.compute_clf_log_suc_prob(grasp_type, latent_voxel, grasp_config)
    #log_prior = grasp_pgm_inf.compute_grasp_log_prior(grasp_type, grasp_config)
    ##print 'clf_log_prob:', clf_log_prob
    #print 'log_prior:', log_prior
    ##obj_prob = clf_log_prob + log_prior 
    #suc_log_prob = grasp_data_file['object_0_grasp_1_inf_suc_prob'][()]
    #print 'suc_log_prob:', suc_log_prob
    #clf_log_prob = suc_log_prob - log_prior 
    #print 'clf_log_prob:', clf_log_prob
    #grasp_data_file.close()



if __name__ == '__main__':
    get_suc_prob()
    #test_suc_prob()
