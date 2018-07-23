import numpy as np
import numpy.random as npr
import time
import sklearn.svm as skl_svm


class KernelHashing:

    def __init__(self,
                 num_hash_functions=1000,
                 num_kernel_computations=100,
                 is_rnd_max_margin=False,
                 num_cores=1,
            ):
        #
        self.set_parameters(
            num_hash_functions=num_hash_functions,
            num_kernel_computations=num_kernel_computations,
            is_rnd_max_margin=is_rnd_max_margin,
            num_cores=num_cores,
        )

    def set_parameters(self,
                       num_hash_functions,
                       num_kernel_computations,
                       is_rnd_max_margin,
                       num_cores,
                    ):
        #
        self.__num_hash_functions__ = num_hash_functions
        self.__num_kernel_computations__ = num_kernel_computations
        self.is_rnd_max_margin = is_rnd_max_margin
        self.num_cores = num_cores
        #
        # keep this False
        self.is_neural_hash = False
        self.is_knn_hashing_partition = False
        #
        self.hash_data_references_pow = None
        self.is_linear_hyperplane_fr_hash = False
        self.__max_new_references__ = 100
        self.__min_bucket_size_fr_sample__ = 10
        self.is_nn_hash_debug = False
        self.is_opt_hash_config = True
        self.opt_hash_config_sample_size = 3000
        self.opt_hash_config_data_ratio = 1.0
        self.is_normalize_kernel_fr_hashcodes = True
        self.__min_bucket_size_fr_sample__ = None
        #
        self.is_error_analysis = False

    def sample_random_subsets(self,
                              num_kernel_computations,
                              is_set2,
                              num_data_per_set
                            ):
        #
        npr.seed(0)
        #
        if num_data_per_set is None:
            assert self.__num_data_per_hashset__ is not None
            num_data_per_set = self.__num_data_per_hashset__
        #
        num_hash_functions = self.__num_hash_functions__
        print 'num_hash_functions', num_hash_functions
        #
        # set of samples for each of the hash functions
        set1_points = -1 * np.ones(shape=(num_hash_functions, num_data_per_set), dtype=np.int64)
        for curr_set_idx in range(num_hash_functions):
            set1_points[curr_set_idx, :] = npr.choice(
                num_kernel_computations,
                num_data_per_set,
                replace=False,
                p=None
            )
        assert np.all(set1_points >= 0)
        #
        if is_set2:
            # set of samples for each of the hash functions
            set2_points = -1 * np.ones(shape=(num_hash_functions, num_data_per_set), dtype=np.int64)
            for curr_set_idx in range(num_hash_functions):
                set2_points[curr_set_idx, :] = npr.choice(
                    num_kernel_computations,
                    num_data_per_set,
                    replace=False,
                    p=None
                )
            assert np.all(set2_points >= 0)
        #
        if is_set2:
            return set1_points, set2_points
        else:
            return set1_points, None

    def compute_kernel(self,
                       path_tuples1,
                       path_tuples2
                    ):
        #
        # dummy kernel code
        # TODO: replace it with real code as per appropriate data
        num_data1 = path_tuples1.shape[0]
        num_data2 = path_tuples1.shape[0]
        K = npr.rand(num_data1, num_data2)
        if num_data1 == num_data2:
            K = 0.5*(K+K.T)
        #
        return K

    def compute_hash_codes(
            self,
            path_tuples,
            path_tuples_kernel_references,
            set1_points,
            set2_points,
            is_return_kernel_matrix=False,
            is_return_Krr=False,
            K_all_reference=None,
            Krr=None
        ):
        #
        num_data = path_tuples.size
        #
        if self.is_rnd_max_margin:
            assert self.is_normalize_kernel_fr_hashcodes
        #
        if K_all_reference is None:
            #
            # computing kernel similarity of each data point w.r.t. the reference points, for the hash functions
            K_all_reference = self.compute_kernel(
                path_tuples1=path_tuples,
                path_tuples2=path_tuples_kernel_references,
            )
            path_tuples = None
        #
        #
        num_hash_functions = set1_points.shape[0]
        num_samples_per_set = set1_points.shape[1]
        # print 'num_samples_per_set', num_samples_per_set
        #
        assert set2_points is not None
        assert set2_points.shape[0] == num_hash_functions
        assert set2_points.shape[1] == num_samples_per_set
        #
        #
        num_samples_per_set_half = int(num_samples_per_set / 2)
        # print 'num_samples_per_set_half', num_samples_per_set_half
        #
        #
        if self.is_rnd_max_margin:
            if Krr is None:
                Krr = self.compute_kernel(path_tuples1=path_tuples_kernel_references,
                                          path_tuples2=path_tuples_kernel_references
                                        )
            #
            svm_clf_objs_list = []
            #
            # assert (num_samples_per_set%2) == 0
            #
            start_svm_clf_time = time.time()
            # build SVM classifiers
            for curr_hash_idx in range(num_hash_functions):
                start_time_rmm_hash = time.time()
                print 'Learning RMM hash function  ...'
                #
                # curr_points = np.concatenate((set1_points[curr_hash_idx, :], set2_points[curr_hash_idx, :]))
                #
                curr_points = set1_points[curr_hash_idx, :]
                curr_Krr = Krr[curr_points, :]
                curr_Krr = curr_Krr[:, curr_points]
                curr_points = None
                #
                # print '{}'.format(curr_Krr.mean()),
                assert curr_Krr.mean() > 0
                #
                curr_labels = np.concatenate((-1*np.ones(num_samples_per_set_half), np.ones(num_samples_per_set-num_samples_per_set_half)))
                #
                curr_svm_clf_obj = skl_svm.SVC(
                    C=1.0,
                    kernel='precomputed',
                    probability=False,
                    verbose=False,
                    random_state=0,
                    # class_weight='balanced'
                )
                #
                curr_svm_clf_obj.fit(
                    curr_Krr,
                    curr_labels
                )
                #
                curr_svm_clf_obj.predict(curr_Krr)
                #
                svm_clf_objs_list.append(curr_svm_clf_obj)
                print 'Time to build RMM hash function', time.time()-start_time_rmm_hash
                start_time_rmm_hash = None
            #
            # time_svm_clf_train = time.time() - start_svm_clf_time
            # print '{}'.format(format(time_svm_clf_train, '.2f')),
            # time_svm_clf_train = None
            print 'Time to train the SVM classifiers for constructing hash functions', time.time() - start_svm_clf_time
        else:
            pass
            # assert Krr is None
        #
        #
        path_tuples_kernel_references = None
        #
        #
        hash_codes = np.zeros(shape=(num_data, num_hash_functions), dtype=np.bool)
        hash_code_data_idx_map = {}
        hash_codes_tuples_list = []
        #
        #
        if self.is_rnd_max_margin:
            for curr_hash_idx in range(num_hash_functions):
                #
                curr_set_kernel = K_all_reference[:, set1_points[curr_hash_idx, :]]
                #
                curr_inferred_labels = svm_clf_objs_list[curr_hash_idx].predict(curr_set_kernel)
                assert curr_inferred_labels.size == num_data
                #
                curr_hashcode_bit = np.zeros(num_data, dtype=bool)
                curr_hashcode_bit[np.where(curr_inferred_labels == 1)] = 1
                hash_codes[:, curr_hash_idx] = curr_hashcode_bit
        else:
            if self.is_knn_hashing_partition:
                for curr_hash_idx in range(num_hash_functions):
                    curr_set1_kernel_max = K_all_reference[:, set1_points[curr_hash_idx, :num_samples_per_set_half]].max(1)
                    assert curr_set1_kernel_max.size == num_data
                    curr_set2_kernel_max = K_all_reference[:, set1_points[curr_hash_idx, num_samples_per_set_half:]].max(1)
                    assert curr_set2_kernel_max.size == num_data
                    #
                    curr_hashcode_bit = np.zeros(num_data, dtype=bool)
                    #
                    curr_hashcode_bit[np.where(curr_set1_kernel_max <= curr_set2_kernel_max)] = 1
                    hash_codes[:, curr_hash_idx] = curr_hashcode_bit
            else:
                for curr_hash_idx in range(num_hash_functions):
                    curr_set1_kernel_max = K_all_reference[:, set1_points[curr_hash_idx, :]].max(1)
                    assert curr_set1_kernel_max.size == num_data
                    curr_set2_kernel_max = K_all_reference[:, set2_points[curr_hash_idx, :]].max(1)
                    assert curr_set2_kernel_max.size == num_data
                    #
                    curr_hashcode_bit = np.zeros(num_data, dtype=bool)
                    #
                    curr_hashcode_bit[np.where(curr_set1_kernel_max <= curr_set2_kernel_max)] = 1
                    hash_codes[:, curr_hash_idx] = curr_hashcode_bit
        #
        #
        for curr_data_idx in range(num_data):
            curr_hash_binary_code = hash_codes[curr_data_idx, :]
            curr_hash_binary_code_tuple = tuple(curr_hash_binary_code.tolist())
            hash_codes_tuples_list.append(curr_hash_binary_code_tuple)
            if curr_hash_binary_code_tuple not in hash_code_data_idx_map:
                hash_code_data_idx_map[curr_hash_binary_code_tuple] = []
            hash_code_data_idx_map[curr_hash_binary_code_tuple].append(curr_data_idx)
            curr_hash_binary_code_tuple = None
        #
        #
        if self.is_error_analysis:
            print 'hash_code_data_idx_map', hash_code_data_idx_map
        #
        #
        set1_points = None
        set2_points = None
        #
        #
        assert np.all(hash_codes >= 0)
        np.save('./hash_codes', hash_codes)
        if self.is_error_analysis:
            print 'hash_codes', hash_codes
        #
        if is_return_Krr:
            if is_return_kernel_matrix:
                return hash_codes, hash_code_data_idx_map, hash_codes_tuples_list, K_all_reference, Krr
            else:
                return hash_codes, hash_code_data_idx_map, hash_codes_tuples_list, Krr
        else:
            if is_return_kernel_matrix:
                return hash_codes, hash_code_data_idx_map, hash_codes_tuples_list, K_all_reference
            else:
                return hash_codes, hash_code_data_idx_map, hash_codes_tuples_list

    def random_assign_path_tuple_kernel_references(self,
                                                   path_tuples=None,
                                                   seed_val=None
                                                ):
        #
        if seed_val is not None:
            npr.seed(seed_val)
        #
        if path_tuples is None:
            if self.path_tuples_arr_unique is None:
                unique_idx = self.__get_unique_train_idx__(self.path_tuples_arr,
                                                           np.arange(self.path_tuples_arr.size, dtype=np.int))
                self.path_tuples_arr_unique = self.path_tuples_arr[unique_idx]
            #
            path_tuples = self.path_tuples_arr_unique
        #
        num_kernel_computations = self.__num_kernel_computations__
        #
        data_idx = np.arange(path_tuples.size, dtype=np.int)
        #
        # sampling the complete set of reference points used for computing the hash functions
        kernel_references_idx = npr.choice(
            data_idx,
            num_kernel_computations,
            replace=False,
            p=None
        )
        #
        print 'kernel_references_idx', kernel_references_idx
        #
        path_tuples_kernel_references = path_tuples[kernel_references_idx]
        kernel_references_idx = None
        path_tuples = None
        self.path_tuples_kernel_references = path_tuples_kernel_references
        #
        return path_tuples_kernel_references

    def optimize_hashing_parameters_rf(self,
                                       path_tuples_arr_train,
                                       labels_train,
                                       path_tuples_kernel_references,
                                       K_all_reference,
                                       Krr,
                                       is_bagging_only=False,
                                       num_mi_lb_trials=1,
                                    ):
        #
        sample_size = min(self.opt_hash_config_sample_size, path_tuples_arr_train.size)
        #
        assert K_all_reference is not None
        assert Krr is not None
        #
        # assert not self.is_rnd_max_margin
        assert not self.is_neural_hash
        assert not self.is_knn_hashing_partition
        #
        if self.is_rnd_max_margin:
            assert self.is_normalize_kernel_fr_hashcodes
        #
        npr.seed(0)
        self.__reset_buffers__()
        #
        _, set1_points, set2_points = self.random_sample_hash_functions(
            is_set2=True
        )
        #
        assert K_all_reference.shape[0] == path_tuples_arr_train.size
        assert K_all_reference.shape[1] == path_tuples_kernel_references.size
        assert Krr.shape[0] == Krr.shape[1] == path_tuples_kernel_references.size
        #
        org_num_hash_functions = self.__num_hash_functions__
        org_is_rnd_max_margin = self.is_rnd_max_margin
        #
        mi_lb_compute_sample_size = int(sample_size/num_mi_lb_trials)
        #
        max_mi_lb = -1e100
        max_mi_lb_config = {}
        #
        # hash_algos_list = ['kulis', 'rmm', 'rknn']
        hash_algos_list = ['rmm']
        # hash_algos_list = ['rknn']
        #
        for hash_algo in hash_algos_list:
            #
            if hash_algo in ['rmm', 'kulis']:
                if hash_algo == 'rmm':
                    self.is_rnd_max_margin = True
                    alpha_list = [6, 8, 10, 16, 20, 30, 40,]
                else:
                    self.is_rnd_max_margin = False
                    alpha_list = [3, 4, 5, 8, 10, 15, 20]
            else:
                self.is_rnd_max_margin = False
                alpha_list = [1, 2, 3, 4, 5, 8, 10, 15, 20]
            #
            for curr_alpha in alpha_list:
                #
                if curr_alpha >= path_tuples_kernel_references.size:
                    continue
                #
                hashbits_list = [100, 300, 1000,]
                # hashbits_list = [300, 1000,]
                #
                for curr_num_hash_bits in hashbits_list:
                    #
                    self.__num_hash_functions__ = curr_num_hash_bits
                    #
                    _, set1_points, set2_points = self.random_sample_hash_functions(
                            is_set2=True,
                            num_data_per_set=curr_alpha,
                    )
                    #
                    #
                    curr_samples_idx = npr.choice(path_tuples_arr_train.size, size=sample_size)
                    curr_path_tuples_train = path_tuples_arr_train[curr_samples_idx]
                    assert curr_path_tuples_train.size == sample_size
                    curr_labels_train = labels_train[curr_samples_idx]
                    assert curr_labels_train.size == sample_size
                    curr_K_all_reference = K_all_reference[curr_samples_idx, :]
                    assert curr_K_all_reference.shape[0] == sample_size
                    #
                    start_time_hashcodes_rnd = time.time()
                    if hash_algo == 'kulis':
                        curr_hashcodes_train, _, _ = self.compute_hash_codes_linear_hyperplane(
                            curr_path_tuples_train,
                            path_tuples_kernel_references,
                            set1_points,
                            K_all_reference=curr_K_all_reference,
                            K=Krr,
                        )
                    else:
                        assert hash_algo in ['rmm', 'rknn']
                        curr_hashcodes_train, _, _ = self.compute_hash_codes(
                            curr_path_tuples_train,
                            path_tuples_kernel_references,
                            set1_points,
                            set2_points,
                            is_return_kernel_matrix=False,
                            K_all_reference=curr_K_all_reference,
                            Krr=Krr,
                        )
                    print 'Time to compute hashcodes', time.time() - start_time_hashcodes_rnd
                    #
                    num_trees_list = [1, 2, 3, 4, 5, 6, 8, 10, 25, 100, 250]
                    #
                    for curr_num_trees in num_trees_list:
                        #
                        # None value means, is_bagging is True
                        # curr_hash_bits_subset_list = [30]
                        # curr_hash_bits_subset_list = [10, 20, 30, 50]
                        # curr_hash_bits_subset_list = [None, 5, 10, 15, 20, 30]
                        # curr_hash_bits_subset_list = [None]
                        #
                        if is_bagging_only:
                            curr_hash_bits_subset_list = [None]
                        else:
                            # curr_hash_bits_subset_list = [None, 5, 10, 15, 20, 30]
                            curr_hash_bits_subset_list = [None]
                        #
                        for curr_hash_bits_subset in curr_hash_bits_subset_list:
                            #
                            print '..............................................'
                            #
                            start_time_mi_lb_compute = time.time()
                            #
                            curr_mi_lb = 0.0
                            #
                            for curr_mi_lb_trial_idx in range(num_mi_lb_trials):
                                #
                                curr_trial_sample_idx = npr.choice(sample_size, size=mi_lb_compute_sample_size)
                                curr_hashcodes_train_trial = curr_hashcodes_train[curr_trial_sample_idx, :]
                                curr_labels_train_trial = curr_labels_train[curr_trial_sample_idx]
                                #
                                if curr_hash_bits_subset is None:
                                    curr_mi_lb_trial = self.compute_rf_mutual_information_lower_bound(
                                            curr_hashcodes=curr_hashcodes_train_trial,
                                            curr_labels=curr_labels_train_trial,
                                            curr_rf_num_trees=curr_num_trees,
                                            is_bagging_mi_lb=True,
                                            # num_cores=self.num_cores,
                                    )
                                else:
                                    curr_mi_lb_trial = self.compute_rf_mutual_information_lower_bound(
                                            curr_hashcodes=curr_hashcodes_train_trial,
                                            curr_labels=curr_labels_train_trial,
                                            curr_rf_num_trees=curr_num_trees,
                                            curr_hashcode_bits=curr_hash_bits_subset,
                                            is_bagging_mi_lb=False,
                                            is_rnd_subselect_bits=True,
                                    )
                                #
                                curr_mi_lb += curr_mi_lb_trial
                            #
                            curr_mi_lb /= num_mi_lb_trials
                            #
                            print 'Time to compute, MI LB', time.time()-start_time_mi_lb_compute
                            start_time_mi_lb_compute = None
                            #
                            #
                            print 'curr_mi_lb', curr_mi_lb
                            #
                            curr_config = {
                                            'alpha': curr_alpha,
                                            'num_trees': curr_num_trees,
                                            'num_bits': curr_num_hash_bits,
                                            'num_bits_subselect': curr_hash_bits_subset,
                                            'hash_algo': hash_algo,
                                        }
                            #
                            #
                            print curr_config
                            print curr_mi_lb
                            #
                            if curr_mi_lb > max_mi_lb:
                                max_mi_lb = curr_mi_lb
                                max_mi_lb_config = curr_config
                                print '+++++++++++++++++++++++++++'
                                print 'max_mi_lb', max_mi_lb
                                print 'max_mi_lb_config', max_mi_lb_config
        #
        self.__num_hash_functions__ = org_num_hash_functions
        self.is_rnd_max_margin = org_is_rnd_max_margin
        #
        return max_mi_lb_config

    def compute_rf_mutual_information_lower_bound(self,
                                                  curr_hashcodes,
                                                  curr_labels,
                                                  curr_rf_num_trees=10,
                                                  num_cores=1,
                                                  curr_hashcode_bits=30,
                                                  is_bagging_mi_lb=None,
                                                  is_rnd_subselect_bits=False,
                                                ):
        #
        print curr_hashcodes.shape
        num_data = curr_labels.size
        #
        if is_bagging_mi_lb is None:
            if self.is_bagging_mi_lb is not None:
                is_bagging_mi_lb = self.is_bagging_mi_lb
            else:
                is_bagging_mi_lb = False
        #
        if is_bagging_mi_lb:
            rf_num_trees = curr_rf_num_trees
            _, inferred_labels_prob = self.random_forest_hash_codes(
                curr_hashcodes,
                curr_hashcodes,
                curr_labels,
                rf_num_trees,
                is_infer_prob=True,
                is_subselect_features=False,
                curr_hashcode_bits=curr_hashcodes.shape[1],
                is_mapping=True,
                num_cores=num_cores,
            )
        else:
            rf_num_trees = 1
            num_trials_rf_subselect = curr_rf_num_trees
            #
            inferred_labels_prob = np.zeros(num_data)
            #
            num_hashcode_bits_all = curr_hashcodes.shape[1]
            #
            if not is_rnd_subselect_bits:
                assert num_hashcode_bits_all >= (curr_hashcode_bits*num_trials_rf_subselect)
            #
            for curr_subselect_trial in range(num_trials_rf_subselect):
                #
                if is_rnd_subselect_bits:
                    curr_hashcode_bits_idx = npr.choice(num_hashcode_bits_all, size=curr_hashcode_bits, replace=False)
                else:
                    curr_hashcode_bits_idx = range((curr_subselect_trial*curr_hashcode_bits), ((curr_subselect_trial+1)*curr_hashcode_bits))
                #
                curr_hashcodes_sel = curr_hashcodes[:, curr_hashcode_bits_idx]
                #
                _, curr_inferred_labels_prob = self.random_forest_hash_codes(
                    curr_hashcodes_sel,
                    curr_hashcodes_sel,
                    curr_labels,
                    rf_num_trees,
                    is_infer_prob=True,
                    is_subselect_features=False,
                    curr_hashcode_bits=curr_hashcode_bits,
                    is_mapping=True,
                    num_cores=num_cores,
                )
                inferred_labels_prob += curr_inferred_labels_prob
                curr_inferred_labels_prob = None
            #
            inferred_labels_prob /= num_trials_rf_subselect
        #
        curr_neg_labels_idx = np.where(curr_labels == 0)[0]
        inferred_labels_prob[curr_neg_labels_idx] = 1 - inferred_labels_prob[curr_neg_labels_idx]
        #
        curr_mi_lb = np.log(inferred_labels_prob + 1e-2).mean()
        #
        # print 'curr_mi_lb', curr_mi_lb
        #
        return curr_mi_lb

    def random_forest_hash_codes(
            self,
            hash_codes_tuples_list_train,
            hash_codes_tuples_list_test,
            train_labels,
            curr_rf_num_trees,
            is_infer_prob=False,
            is_subselect_features=False,
            curr_hashcode_bits=None,
            is_mapping=False,
            num_cores=None,
            is_rf_classifier_return=False,
            is_test=True,
    ):
        #
        #
        if is_test:
            assert hash_codes_tuples_list_test is not None
        #
        #
        if num_cores is None:
            num_cores = self.num_cores
        #
        if is_mapping:
            X_train_org = hash_codes_tuples_list_train
            if is_test:
                X_test_org = hash_codes_tuples_list_test
            # num_hash_code_bits = X_train_org.shape[1]
            # assert num_hash_code_bits == X_test_org.shape[1]
        else:
            X_train_org = self.map_hashcodes_list_to_array(hash_codes_tuples_list_train)
            if is_test:
                X_test_org = self.map_hashcodes_list_to_array(hash_codes_tuples_list_test)
        #
        # print 'X_train_org.shape', X_train_org.shape
        # print 'X_test_org.shape', X_test_org.shape
        #
        #
        if is_subselect_features:
            num_bits = X_train_org.shape[1]
            if is_test:
                assert num_bits == X_test_org.shape[1]
            curr_bits_sel = npr.choice(num_bits, size=curr_hashcode_bits, replace=False)
            X_train_org = X_train_org[:, curr_bits_sel]
            if is_test:
                X_test_org = X_test_org[:, curr_bits_sel]
        #
        #
        # also try out boosting trees
        #
        # Random Forests
        random_forest_classifier = sklearn.ensemble.RandomForestClassifier(
                                        n_estimators=curr_rf_num_trees,
                                        n_jobs=num_cores,
                                        class_weight='balanced',
                                        criterion='gini',
                                    )
        random_forest_classifier.fit(X_train_org, train_labels)
        #
        if is_test:
            inferred_labels = random_forest_classifier.predict(X_test_org)
            #
            if is_infer_prob:
                inferred_labels_prob = random_forest_classifier.predict_proba(X_test_org)
                # print random_forest_classifier.classes_
                inferred_labels_prob = inferred_labels_prob[:, 1]
                # print 'inferred_labels_prob.shape', inferred_labels_prob.shape
            #
            # precision, recall, f1, num_true_pos, num_false_pos, num_true_neg, num_false_neg = \
            #     self.__eval_error_on_inferred_labels__(gold_labels=test_labels,
            #                                            inferred_labels=inferred_labels)
            # print 'P{}, R{}: F1{}'.format(precision, recall, f1)
            # print 'TP: {}, FP: {}, TN: {}, FN: {}'.format(num_true_pos, num_false_pos, num_true_neg, num_false_neg)
            #
            if is_infer_prob:
                if is_rf_classifier_return:
                    return inferred_labels, inferred_labels_prob, random_forest_classifier
                else:
                    return inferred_labels, inferred_labels_prob
            else:
                if is_rf_classifier_return:
                    return inferred_labels, random_forest_classifier
                else:
                    return inferred_labels
        else:
            return random_forest_classifier


if __name__ == '__main__':
    #
    num_kernel_computations = 100
    num_hash_functions = 100
    # If RMM is False, RkNN is used, i.e. Random K Nearest Neighbors Hashing.
    is_rnd_max_margin = True
    #
    kh_obj = KernelHashing(
        num_hash_functions=num_hash_functions,
        num_kernel_computations=num_kernel_computations,
        is_rnd_max_margin=is_rnd_max_margin,
    )
    #
    # TODO: replace this with your own feature vectors
    # the current code is such that an element of an array should be an object .
    # if representing coordinates in robotics problems as features,
    # represent each coordinate vector as an object rather than having  high dimensianal arrays. This is for uniformity of code over other types of data such as structures (strings).
    path_tuples_arr = np.array(['dummy']*10000)
    print path_tuples_arr.size
    # path_tuples_arr = np.load('./path_tuples_data_fr_test.npy')
    # print path_tuples_arr.size
    # path_tuples_arr = path_tuples_arr[npr.choice(path_tuples_arr.size, 3000)]
    #
    path_tuples_kernel_references = kh_obj.random_assign_path_tuple_kernel_references(
        path_tuples=path_tuples_arr,
        seed_val=0
    )
    #
    print 'sampling hash function related random subsets'
    set1_points, set2_points = kh_obj.sample_random_subsets(
        num_kernel_computations=num_kernel_computations,
        is_set2=True,
        num_data_per_set=20,
    )
    #
    print 'computing hash codes ...'
    hash_codes, _, _ = kh_obj.compute_hash_codes(
        path_tuples=path_tuples_arr,
        path_tuples_kernel_references=path_tuples_kernel_references,
        set1_points=set1_points,
        set2_points=set2_points,
    )
    #
    print hash_codes
