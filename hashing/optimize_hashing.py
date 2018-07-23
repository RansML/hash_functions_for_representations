class OptimizeHashing:

    def __init__(self):
        pass

    def optimize_references_greedy_fr_rf(
            self,
            is_dump=False,
            sample_size=100,
            suffix='',
            data_set=None,
            initial_ref_set_size=10,
            max_ref_set_size=100,
            ref_sample_size=1000,
            seed_val=0
    ):
        # is_hash_not is used for evaluation of hashing against the standard approaches
        #
        # Jan 4, 2018
        #
        npr.seed(0)
        #
        self.__reset_buffers__()
        #
        #
        if self.is_rnd_max_margin:
            assert self.is_normalize_kernel_fr_hashcodes
        #
        #
        path_tuples_arr = self.path_tuples_arr
        #
        if data_set is not None:
            if data_set == 'aimed':
                self.process_aimed_data_fr_protein_path_kernel_clf()
                path_tuples_arr = self.path_tuples_arr
            else:
                dataset_idx = self.__get_specific_dataset_idx__(
                    amr_graphs=path_tuples_arr,
                    data_set_name=data_set
                )
                path_tuples_arr = path_tuples_arr[dataset_idx]
                print 'dataset_idx', dataset_idx
                print 'path_tuples_arr.shape', path_tuples_arr.shape
                dataset_idx = None
        #
        if self.__is_unique_paths__:
            unique_idx = self.__get_unique_train_idx__(path_tuples_arr, np.arange(path_tuples_arr.size, dtype=np.int))
            path_tuples_arr = path_tuples_arr[unique_idx]
            unique_idx = None
        #
        labels = self.get_labels_frm_path_tuples_data(path_tuples_arr)
        #
        self.__num_kernel_computations__ = initial_ref_set_size
        self.path_tuples_kernel_references = None
        self.random_assign_path_tuple_kernel_references(
            path_tuples=path_tuples_arr,
            seed_val=seed_val
        )
        path_tuples_kernel_references = self.path_tuples_kernel_references
        #
        if (self.is_bagging_mi_lb is not None) and self.is_bagging_mi_lb:
            self.__num_hash_functions__ = 100
        else:
            self.__num_hash_functions__ = 300
        #
        _, set1_points, set2_points = self.random_sample_hash_functions(
            path_tuples=path_tuples_arr,
            is_set2=True
        )
        print 'path_tuples_kernel_references.shape', path_tuples_kernel_references.shape
        #
        #
        for curr_ref_idx in range(max_ref_set_size):
            #
            start_time = time.time()
            #
            #
            if curr_ref_idx == path_tuples_kernel_references.size:
                new_ref_data_idx = rnd.randint(0, path_tuples_arr.size - 1)
                path_tuples_kernel_references = np.append(path_tuples_kernel_references,
                                                          path_tuples_arr[new_ref_data_idx])
                print 'new_ref_data_idx', new_ref_data_idx
                new_ref_data_idx = None
                assert path_tuples_kernel_references.size == (curr_ref_idx + 1)
                #
                #
                if not self.is_rnd_max_margin:
                    num_data_per_set = max(int(self.__num_samples_fr_hash_ratio__ * path_tuples_kernel_references.size), 1)
                    set1_points, set2_points =\
                        self.sample_random_subsets(
                            path_tuples_kernel_references.size,
                            True,
                            num_data_per_set=num_data_per_set
                        )
            else:
                assert curr_ref_idx < path_tuples_kernel_references.size
            #
            #
            # optimize (curr_idx)th reference point
            #
            # subselect data for computing mutual information
            subselect_data_idx_fr_mi = npr.choice(path_tuples_arr.size, size=sample_size)
            curr_path_tuples_arr = path_tuples_arr[subselect_data_idx_fr_mi]
            print 'curr_path_tuples_arr.shape', curr_path_tuples_arr.shape
            curr_labels = labels[subselect_data_idx_fr_mi]
            subselect_data_idx_fr_mi = None
            #
            #
            if self.is_rnd_max_margin:
                Krr = __compute_kernel_matrix_parallel__(
                    self,
                    path_tuples_kernel_references,
                    path_tuples_kernel_references,
                    is_sparse=True,
                    is_normalize=True
                )
                #
                if Krr.size == 0:
                    Krr = np.zeros(shape=Krr.shape)
                else:
                    Krr = Krr.toarray()
            else:
                Krr = None
            #
            #
            _, _, _, kernel_references = self.compute_hash_codes(
                                                                    curr_path_tuples_arr,
                                                                    path_tuples_kernel_references,
                                                                    set1_points,
                                                                    set2_points,
                                                                    is_return_kernel_matrix=True
                                                                    )
            #
            #
            ref_candidates_idx = npr.choice(path_tuples_arr.size, size=ref_sample_size)
            path_tuples_arr_fr_changes = path_tuples_arr[ref_candidates_idx]
            path_tuples_arr_fr_changes = np.append(path_tuples_arr_fr_changes, path_tuples_kernel_references[curr_ref_idx])
            ref_candidates_idx = None
            #
            #
            kernel_references_changes = __compute_kernel_matrix_parallel__(
                self,
                curr_path_tuples_arr,
                path_tuples_arr_fr_changes,
                is_sparse=True,
                is_normalize=self.is_normalize_kernel_fr_hashcodes
            )
            kernel_references_changes = kernel_references_changes.toarray()
            #
            # path_tuples_arr_fr_changes = None
            #
            #
            if self.is_rnd_max_margin:
                Krr_changes = __compute_kernel_matrix_parallel__(
                    self,
                    path_tuples_kernel_references,
                    path_tuples_arr_fr_changes,
                    is_sparse=True,
                    is_normalize=True
                )
                Krr_changes = Krr_changes.toarray()
            else:
                Krr_changes = None
            #
            #
            mi_lb_arr = self.compute_mutual_information_lower_bound_parallel_wrapper(
                path_tuples_arr_fr_changes,
                kernel_references_changes,
                Krr_changes,
                curr_path_tuples_arr,
                set1_points,
                set2_points,
                curr_labels,
                kernel_references,
                Krr,
                curr_ref_idx
            )
            #
            #
            max_mi_idx = mi_lb_arr.argmax()
            min_mi_idx = mi_lb_arr.argmin()
            print 'mi_lb_arr', mi_lb_arr
            print 'mi_lb_arr.max()', mi_lb_arr.max()
            print 'max_mi_idx', max_mi_idx
            print 'mi_lb_arr.min()', mi_lb_arr.min()
            print 'mi_lb_arr.mean()', mi_lb_arr.mean()
            #
            path_tuples_kernel_references[curr_ref_idx] = path_tuples_arr_fr_changes[max_mi_idx]
            #
            print 'max:'
            print path_tuples_arr_fr_changes[max_mi_idx]
            #
            print 'min:'
            print path_tuples_arr_fr_changes[min_mi_idx]
            #
            print '{}th optimized in {}'.format(curr_ref_idx, time.time()-start_time)
            #
            #
        #
        self.path_tuples_kernel_references = path_tuples_kernel_references
        self.__num_kernel_computations__ = path_tuples_kernel_references.size
        #
        if is_dump:
            self.__dump__(suffix=suffix)

    def optimize_artificial_labels_hashing_greedy_fr_rf_parallel(self,
                                                                 path_tuples_arr_train,
                                                                 path_tuples_kernel_references,
                                                                 K_all_reference,
                                                                 set1_points,
                                                                 set2_points):
        #
        num_cores = self.num_cores
        num_hash = set1_points.shape[0]
        print 'num_hash', num_hash
        assert num_hash == set2_points.shape[0]
        #
        idx_range_parallel = pc.uniform_distribute_tasks_across_cores(num_hash, num_cores)
        args_tuples_map = {}
        for currCore in range(num_cores):
            args_tuples_map[currCore] = (
                        path_tuples_arr_train,
                        path_tuples_kernel_references,
                        K_all_reference,
                        set1_points[idx_range_parallel[currCore], :],
                        set2_points[idx_range_parallel[currCore], :],
                    )
        #
        pcw_obj = pcw.ParallelComputingWrapper(num_cores=num_cores)
        results_map = pcw_obj.process_method_parallel(
            method=self.optimize_artificial_labels_hashing_greedy_fr_rf,
            args_tuples_map=args_tuples_map,
        )
        #
        for curr_core in range(num_cores):
            curr_result = results_map[curr_core]
            curr_core_set1_points = curr_result[0]
            curr_core_set2_points = curr_result[1]
            assert curr_core_set1_points.shape[1] == curr_core_set2_points.shape[1]
            set1_points[idx_range_parallel[curr_core], :] = curr_core_set1_points
            set2_points[idx_range_parallel[curr_core], :] = curr_core_set2_points
        #
        return set1_points, set2_points

    def optimize_artificial_labels_hashing_greedy_fr_rf_wrapper(
            self,
            path_tuples_arr_train,
            path_tuples_kernel_references,
            K_all_reference,
            set1_points,
            set2_points,
            queue,
    ):
        try:
            set1_points, set2_points = self.optimize_artificial_labels_hashing_greedy_fr_rf(
                path_tuples_arr_train,
                path_tuples_kernel_references,
                K_all_reference,
                set1_points,
                set2_points,
            )
            queue.put([set1_points, set2_points])
        except BaseException as e:
            print 'error in the subprocess (base exception)'
            print e
            queue.put(e)
        except OSError as ee:
            print 'error in the subprocess (os error)'
            print ee
            queue.put(ee)

    def optimize_artificial_labels_hashing_greedy_fr_rf(
            self,
            path_tuples_arr_train,
            path_tuples_kernel_references,
            K_all_reference,
            set1_points,
            set2_points,
    ):
        #
        # May 7, 2018
        #
        sample_size = min(path_tuples_arr_train.size, self.art_label_opt_sample_size)
        #
        assert not self.is_neural_hash
        assert not self.is_knn_hashing_partition
        if self.is_rnd_max_margin:
            assert self.is_normalize_kernel_fr_hashcodes
        #
        npr.seed(0)
        self.__reset_buffers__()
        #
        labels_train = self.get_labels_frm_path_tuples_data(path_tuples_arr_train)
        #
        start_time_hashcodes_rnd = time.time()
        hashcodes_train, _, _ = self.compute_hash_codes(
            path_tuples_arr_train,
            path_tuples_kernel_references,
            set1_points,
            set2_points,
            is_return_kernel_matrix=False,
            K_all_reference=K_all_reference,
        )
        print 'Time to compute hashcodes', time.time()-start_time_hashcodes_rnd
        #
        num_hash = set1_points.shape[0]
        print 'num_hash', num_hash
        assert num_hash == set2_points.shape[0]
        alpha = set1_points.shape[1]
        assert alpha == set2_points.shape[1]
        #
        # num_mcmc_samples = int(alpha ** 2)
        num_mcmc_samples = self.mcmc_sample_size
        num_dt_fr_mi_lb = self.num_dt_fr_mcmc_lkl
        #
        mi_lb_greedy_opt = np.empty(shape=num_hash, dtype=float)
        # sample_size = max(int(path_tuples_arr_train.size * 0.03), sample_size, replace=False)
        # sample_size = int(path_tuples_arr_train.size * 0.01)
        #
        for curr_hash_idx in range(num_hash):
            #
            curr_hash_start_time = time.time()
            #
            print '********************************'
            print 'curr_hash_idx', curr_hash_idx
            #
            curr_samples_idx = npr.choice(path_tuples_arr_train.size, size=sample_size, )
            curr_path_tuples_train = path_tuples_arr_train[curr_samples_idx]
            assert curr_path_tuples_train.size == sample_size
            print 'curr_path_tuples_train', curr_path_tuples_train.size
            #
            curr_K_all_reference = K_all_reference[curr_samples_idx, :]
            assert curr_K_all_reference.shape[0] == sample_size
            #
            curr_hashcodes_train = hashcodes_train[curr_samples_idx, :curr_hash_idx+1]
            # curr_hashcodes_train = hashcodes_train[curr_samples_idx, :]
            assert curr_hashcodes_train.shape[0] == sample_size
            assert curr_hashcodes_train.shape[1] == (curr_hash_idx+1)
            #
            curr_labels_train = labels_train[curr_samples_idx]
            assert curr_labels_train.size == sample_size
            #
            curr_hash_set1_points = set1_points[curr_hash_idx, :]
            assert curr_hash_set1_points.size == alpha
            curr_hash_set2_points = set2_points[curr_hash_idx, :]
            assert curr_hash_set2_points.size == alpha
            #
            # implement an MCMC algorithm, with proposal defined by exchange of two nodes between the sets
            # for efficiency, recompute hashcodes only for current hash functions, passing only one row of set1,  set2
            # for computing MI LB, we can pass all the hash functions
            set1_points_mcmc = np.empty(dtype=curr_hash_set1_points.dtype, shape=(num_mcmc_samples, alpha))
            set2_points_mcmc = np.empty(dtype=curr_hash_set1_points.dtype, shape=(num_mcmc_samples, alpha))
            mi_lb_mcmc = np.empty(shape=num_mcmc_samples, dtype=float)
            #
            start_time = time.time()
            swap_idx_set1_mcmc = npr.randint(low=0, high=alpha, size=num_mcmc_samples)
            swap_idx_set2_mcmc = npr.randint(low=0, high=alpha, size=num_mcmc_samples)
            uniform_rnd_mcmc = npr.uniform(low=0.0, high=1.0, size=num_mcmc_samples)
            print 'Time to obtain random swap indices', time.time()-start_time
            #
            curr_mcmc_idx = 0
            curr_mi_lb = self.compute_rf_mutual_information_lower_bound(
                # curr_hashcodes=curr_hashcodes_train[:, :curr_hash_idx+1],
                curr_hashcodes=curr_hashcodes_train,
                curr_labels=curr_labels_train,
                curr_rf_num_trees=num_dt_fr_mi_lb,
                # num_cores=self.num_cores,
            )
            print 'curr_mi_lb', curr_mi_lb
            mi_lb_mcmc[curr_mcmc_idx] = curr_mi_lb
            set1_points_mcmc[curr_mcmc_idx, :] = curr_hash_set1_points
            set2_points_mcmc[curr_mcmc_idx, :] = curr_hash_set2_points
            curr_mcmc_idx += 1
            #
            while curr_mcmc_idx < num_mcmc_samples:
                #
                print '...................................'
                print 'curr_mcmc_idx', curr_mcmc_idx
                curr_set1_swap_idx = swap_idx_set1_mcmc[curr_mcmc_idx]
                curr_set2_swap_idx = swap_idx_set2_mcmc[curr_mcmc_idx]
                #
                temp = curr_hash_set1_points[curr_set1_swap_idx]
                curr_hash_set1_points[curr_set1_swap_idx] = curr_hash_set2_points[curr_set2_swap_idx]
                curr_hash_set2_points[curr_set2_swap_idx] = temp
                temp = None
                #
                print curr_hash_set1_points
                print curr_hash_set2_points
                #
                start_time_compute_hashcode_vec = time.time()
                curr_hashcode_vector, _, _ = self.compute_hash_codes(
                    curr_path_tuples_train,
                    path_tuples_kernel_references,
                    curr_hash_set1_points.reshape((1, alpha)),
                    curr_hash_set2_points.reshape((1, alpha)),
                    is_return_kernel_matrix=False,
                    K_all_reference=curr_K_all_reference,
                )
                assert curr_hashcode_vector.shape[1] == 1
                curr_hashcodes_train[:, curr_hash_idx] = curr_hashcode_vector[:, 0]
                print 'Time to compute hashcode vector', time.time()-start_time_compute_hashcode_vec
                #
                start_time_compute_mi_lb = time.time()
                curr_mi_lb = self.compute_rf_mutual_information_lower_bound(
                    # curr_hashcodes=curr_hashcodes_train[:, :curr_hash_idx+1],
                    curr_hashcodes=curr_hashcodes_train,
                    curr_labels=curr_labels_train,
                    curr_rf_num_trees=num_dt_fr_mi_lb,
                    # num_cores=self.num_cores,
                )
                print 'curr_mi_lb', curr_mi_lb
                print 'Time to compute mi lb', time.time()-start_time_compute_mi_lb
                #
                if curr_mi_lb >= mi_lb_mcmc[curr_mcmc_idx-1]:
                    acceptance_ratio = 1.0
                else:
                    # todo: this code block should belong to cython since math functions being used
                    # assuming that lkl values can be negative as well
                    acceptance_ratio = math.exp(
                                                -10*float(abs(curr_mi_lb - mi_lb_mcmc[curr_mcmc_idx-1]))
                                                /
                                                abs(mi_lb_mcmc[curr_mcmc_idx-1])
                                            )
                #
                print 'acceptance_ratio', acceptance_ratio
                curr_trial_prob = uniform_rnd_mcmc[curr_mcmc_idx]
                print curr_trial_prob
                #
                if curr_trial_prob > acceptance_ratio:
                    print 'sample rejected'
                    curr_mi_lb = mi_lb_mcmc[curr_mcmc_idx-1]
                    curr_hash_set1_points = set1_points_mcmc[curr_mcmc_idx-1, :]
                    curr_hash_set2_points = set2_points_mcmc[curr_mcmc_idx-1, :]
                #
                mi_lb_mcmc[curr_mcmc_idx] = curr_mi_lb
                set1_points_mcmc[curr_mcmc_idx, :] = curr_hash_set1_points
                set2_points_mcmc[curr_mcmc_idx, :] = curr_hash_set2_points
                #
                curr_mcmc_idx += 1
            #
            print 'mi_lb_mcmc.mean()', mi_lb_mcmc.mean()
            print 'mi_lb_mcmc.std()', mi_lb_mcmc.std()
            print 'mi_lb_mcmc.min()', mi_lb_mcmc.min()
            print 'mi_lb_mcmc.max()', mi_lb_mcmc.max()
            #
            max_mi_lb_mcmc_sample_idx = mi_lb_mcmc.argmax()
            opt_hash_set1_points = set1_points_mcmc[max_mi_lb_mcmc_sample_idx, :]
            set1_points_mcmc = None
            set1_points[curr_hash_idx, :] = opt_hash_set1_points
            opt_hash_set2_points = set2_points_mcmc[max_mi_lb_mcmc_sample_idx, :]
            set2_points_mcmc = None
            set2_points[curr_hash_idx, :] = opt_hash_set2_points
            max_mi_lb_mcmc_sample_idx = None
            #
            opt_hashcode_vector, _, _ = self.compute_hash_codes(
                path_tuples_arr_train,
                path_tuples_kernel_references,
                opt_hash_set1_points.reshape((1, alpha)),
                opt_hash_set2_points.reshape((1, alpha)),
                is_return_kernel_matrix=False,
                K_all_reference=K_all_reference,
            )
            assert opt_hashcode_vector.shape[1] == 1
            opt_hash_set1_points = None
            opt_hash_set2_points = None
            #
            hashcodes_train[:, curr_hash_idx] = opt_hashcode_vector[:, 0]
            curr_mi_lb = self.compute_rf_mutual_information_lower_bound(
                curr_hashcodes=hashcodes_train[:, :curr_hash_idx + 1],
                # curr_hashcodes=hashcodes_train,
                curr_labels=labels_train,
                curr_rf_num_trees=num_dt_fr_mi_lb,
                # num_cores=self.num_cores,
            )
            mi_lb_greedy_opt[curr_hash_idx] = curr_mi_lb
            print 'mi_lb_greedy_opt', mi_lb_greedy_opt[:curr_hash_idx+1]
            print 'Time to optimize for current hash function {}'.format(curr_hash_idx), time.time()-curr_hash_start_time
        #
        return set1_points, set2_points

    def build_hash_functions_large_set_frm_unlabeled_data(
            self,
            path_tuples_arr_unlabeled,
            num_data_per_set=10,
            seed_val=0,
            num_hash_functions=1000,
            num_kernel_computations=3000,
    ):
        #
        # Mar 29, 2018
        #
        npr.seed(0)
        self.__reset_buffers__()
        #
        assert not self.is_rnd_max_margin,\
            'implementation is assumed for RkNN hashing as it is computationally efficient, stable, and purely data-driven; others hash function implementations' \
            ' may be also be applicable with minimal changes '
        assert not self.is_knn_hashing_partition
        assert not self.is_neural_hash
        #
        self.__num_hash_functions__ = num_hash_functions
        self.__num_kernel_computations__ = num_kernel_computations
        self.path_tuples_kernel_references = None
        self.random_assign_path_tuple_kernel_references(
            path_tuples=path_tuples_arr_unlabeled,
            seed_val=seed_val
        )
        self.path_tuples_kernel_references_superset = self.path_tuples_kernel_references
        _, set1_points, set2_points = self.random_sample_hash_functions(
            path_tuples=None,
            is_set2=True,
            num_data_per_set=num_data_per_set,
        )
        self.set1_points_superset = set1_points
        self.set2_points_superset = set2_points

    def optimal_select_hash_functions_semisupervised(self,
                                                         path_tuples_arr_train,
                                                         labels_train,
                                                         path_tuples_arr_test,
                                                         num_hash_fr_sel=100,
                                                         trn_sample_size=100,
                                                         trn_sample_ratio=0.03,
                                                         tst_sample_size=100,
                                                         tst_sample_ratio=0.1,
                                                     ):
        #
        # 100k size of reference set approx.
        assert self.path_tuples_kernel_references_superset is not None
        assert self.set1_points_superset is not None
        assert self.set2_points_superset is not None
        #
        #
        # assert path_tuples_arr_train.size >= trn_sample_size
        # randomly sample training and test subsets
        sample_size_train = min(max(trn_sample_size, int(path_tuples_arr_train.size*trn_sample_ratio)), path_tuples_arr_train.size)
        rnd_train_idx = npr.choice(path_tuples_arr_train.size, size=sample_size_train, replace=False)
        path_tuples_arr_train = path_tuples_arr_train[rnd_train_idx]
        labels_train = labels_train[rnd_train_idx]
        num_train = path_tuples_arr_train.size
        assert num_train == labels_train.size
        #
        #
        # if path_tuples_arr_test.size >= 30:
        sample_size_test = min(max(tst_sample_size, int(path_tuples_arr_test.size*tst_sample_ratio)), path_tuples_arr_test.size)
        path_tuples_arr_test = path_tuples_arr_test[npr.choice(path_tuples_arr_test.size, size=sample_size_test, replace=False)]
        #     is_test_set_fr_opt = True
        # else:
        #     is_test_set_fr_opt = False
        #
        zeros_idx = np.where(labels_train == 0)[0]
        print 'zeros_idx', zeros_idx.shape
        ones_idx = np.where(labels_train == 1)[0]
        print 'ones_idx', ones_idx.shape
        zeros_ratio = zeros_idx.size/float(num_train)
        ones_ratio = ones_idx.size/float(num_train)
        #
        #
        # compute hashcodes for random subset of training and test sets together, for higher efficieny on kernel computations
        path_tuples_arr = np.concatenate((path_tuples_arr_train, path_tuples_arr_test))
        hashcodes_train_test, _, _ = self.compute_hash_codes(
            path_tuples_arr,
            self.path_tuples_kernel_references_superset,
            self.set1_points_superset,
            self.set2_points_superset,
            is_return_kernel_matrix=False
        )
        hashcodes_train= hashcodes_train_test[:num_train, :]
        print 'hashcodes_train.shape', hashcodes_train.shape
        #
        num_bits = hashcodes_train.shape[1]
        assert self.set1_points_superset.shape[0] == num_bits
        assert self.set2_points_superset.shape[0] == num_bits
        hash_indices_fr_mi_compute = np.arange(num_bits, dtype=int)
        #
        hashcodes_train_zeros = hashcodes_train[zeros_idx, :]
        hashcodes_train_ones = hashcodes_train[ones_idx, :]
        hashcodes_train = None
        print 'hashcodes_train_zeros.shape', hashcodes_train_zeros.shape
        print 'hashcodes_train_ones.shape', hashcodes_train_ones.shape
        #
        #
        # if is_test_set_fr_opt:
        #     # compute hashcodes for random subset of test data
        #     hashcodes_test, _, _ = self.compute_hash_codes(
        #         path_tuples_arr_test,
        #         self.path_tuples_kernel_references_superset,
        #         self.set1_points_superset,
        #         self.set2_points_superset,
        #         is_return_kernel_matrix=False
        #     )
        #     print 'hashcodes_test.shape', hashcodes_test.shape
        #     assert num_bits == hashcodes_test.shape[1]
        # else:
        #     hashcodes_test = None
        #
        #
        # compute marginal entropy for each hash code bit for training data
        e_trn_zeros = self.compute_marginal_entropies(hashcodes_train_zeros)
        e_trn_ones = self.compute_marginal_entropies(hashcodes_train_ones)
        #
        # if is_test_set_fr_opt:
        # compute marginal entropy for each hashcode bit for test data
        e_test = self.compute_marginal_entropies(hashcodes_train_test)
        # else:
        #     e_test = None
        #
        #
        sel_idx = []
        #
        # greedy selection of hash code bits (O(HK^2)) (K ~ 100, H~10000)
        # compute TC(C;Z) on training data, for each choice of bit, along with selected ones
        # compute TC(C;Z) on test data, for each choice of bit, along with selected ones
        # compute H(C) - H(C|Y)
        # select bit with maximum objective.
        for curr_greedy_idx in range(num_hash_fr_sel):
            #
            mi_bound = self.compute_semisupervised_mi_approximation_wrapper(sel_idx,
                                                                     hash_indices_fr_mi_compute,
                                                                     e_trn_zeros,
                                                                     e_trn_ones,
                                                                     e_test,
                                                                     hashcodes_train_zeros,
                                                                     hashcodes_train_ones,
                                                                     hashcodes_train_test,
                                                                     zeros_ratio,
                                                                     ones_ratio,
                                                                 )
            #
            # print 'mi_bound', mi_bound
            #
            # to make sure previously selected hash functions are not selected again
            if sel_idx:
                mi_bound[sel_idx] = -1e5
            #
            print 'mi_bound', mi_bound
            #
            max_mi_sel_idx = mi_bound.argmax()
            print 'max_mi_sel_idx', max_mi_sel_idx
            print 'mi_bound[max_mi_sel_idx]', mi_bound[max_mi_sel_idx]
            sel_idx.append(max_mi_sel_idx)
        #
        print 'sel_idx', sel_idx
        #
        # take subset of hashcode bits (set1, set2)
        # obtain the subset of super set of references (not more than a couple of thousands)
        set1_points = self.set1_points_superset[sel_idx, :]
        set2_points = self.set2_points_superset[sel_idx, :]
        #
        print 'set1_points.shape', set1_points.shape
        print 'set2_points.shape', set2_points.shape
        #
        sel_ref_idx = np.unique(np.concatenate((set1_points.reshape(set1_points.size), set2_points.reshape(set2_points.size))))
        print 'sel_ref_idx', sel_ref_idx
        path_tuples_kernel_references = self.path_tuples_kernel_references_superset[sel_ref_idx]
        # sel_ref_idx = None
        print 'path_tuples_kernel_references.size', path_tuples_kernel_references.size
        #
        set1_points_new = np.zeros(set1_points.shape, dtype=int)
        set2_points_new = np.zeros(set2_points.shape, dtype=int)
        count = -1
        for curr_ref_idx in sel_ref_idx:
            count += 1
            set1_points_new[np.where(set1_points == curr_ref_idx)] = count
            set2_points_new[np.where(set2_points == curr_ref_idx)] = count
        #
        set1_points = set1_points_new
        set1_points_new = None
        set2_points = set2_points_new
        set2_points_new = None
        #
        return path_tuples_kernel_references, set1_points, set2_points

