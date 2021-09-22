import itertools
import logging
import random
from functools import partialmethod

import numpy as np
import pandas as pd

__author__ = 'michael.p.schroeder@gmail.com'

import multiprocessing as mp


class MutExResult(object):
    def __init__(self, coverage, signal, higher_coverage_count, lower_coverage_count, permutations,
                 mean_sim_coverage, stdev_sim_coverage,
                 sample_size, items):
        self.stdev_sim_coverage = stdev_sim_coverage
        self.items = items
        self.sample_size = sample_size
        self.mean_sim_coverage = mean_sim_coverage
        self.permutations = permutations
        self.lower_coverages = lower_coverage_count
        self.higher_coverages = higher_coverage_count
        self.signal = signal
        self.coverage = coverage
        self.signal_coverage_ratio = coverage / signal

        self.mutex_pvalue = higher_coverage_count / permutations
        self.co_occurence_pvalue = lower_coverage_count / permutations
        self.zscore = (coverage - mean_sim_coverage) / stdev_sim_coverage

    def __str__(self):
        return "MuTexResult\n" \
               "  Zscore:                     {}\n" \
               "  Mutual Exclusive p-value:   {}\n" \
               "  Co-occurence p-value:       {}\n" \
               "  Permutations:               {}\n" \
               "  Sample Coverage:            {}\n" \
               "  Signal:                     {}".format(
            self.zscore, self.mutex_pvalue, self.co_occurence_pvalue, self.permutations, self.coverage, self.signal
        )

    def __repr__(self):
        return self.__str__()


class MutEx(object):
    def __init__(self, background: pd.DataFrame, permutations: int = 100):
        """
        :param background: A data frame containing all the observations as binary data 1 and 0 or True and False where
                rows represent observations and columns represent samples.
        :param permutations: how many permutations by default
        :return:
        """
        self.permutations = permutations
        self.background = background
        self.sample_weights = background.apply(sum) / background.apply(sum).pipe(sum)
        self.cummulative_sum = np.cumsum(self.sample_weights)
        self.sample_indices = [x for x in range(0, background.shape[1])]

    def calculate(self, indices: list, n=None, parallel=True, cores=0) -> MutExResult:
        """
        :param indices: A list of indices for which to test the MutEx. The indices refer the the background-data row-ids.
        :return: MutExResult
        """

        if not all([x in self.background.index for x in indices]):
            raise Exception("Not all indices found in background")

        target = self.background.loc[indices]

        coverage = target.apply(max).pipe(sum)
        observation_signal = target.apply(sum, axis=1)
        signal = sum(observation_signal)

        if n == None:
            n = self.permutations

        logging.info("running {} permutations".format(n))

        if not parallel:
            cores = 1
        pool = mp.Pool(processes=mp.cpu_count() if cores < 1 else cores)
        logging.info('permutation with {} cores'.format(pool._processes))
        partial_simul = partialmethod(self._one_permutation)
        simulated_results = pool.starmap(partial_simul.func,
                                         zip(itertools.repeat(coverage, n), itertools.repeat(observation_signal, n)))
        pool.close()  # we are not adding any more processes
        pool.join()  # tell it to wait until all threads are done before going on

        logging.info('calculate result')
        sim_coverages = [x[0] for x in simulated_results]
        higher_coverage = [x[1] for x in simulated_results]
        lower_coverage = [x[2] for x in simulated_results]

        return MutExResult(coverage=coverage, signal=signal,
                           higher_coverage_count=np.sum(higher_coverage),
                           lower_coverage_count=np.sum(lower_coverage), permutations=n,
                           mean_sim_coverage=np.mean(sim_coverages),
                           stdev_sim_coverage=np.std(sim_coverages),
                           sample_size=len(self.sample_weights),
                           items=indices
                           )

    def _one_permutation(self, coverage, observation_signal):
        sim = self._simulate_observations(observation_signal)
        sim_cov = sim.apply(max).pipe(sum)
        higher_cov = sim_cov >= coverage
        lower_cov = sim_cov <= coverage
        return sim_cov, higher_cov, lower_cov

    def _simulate_observations(self, observation_signal):
        simulations = []
        for observations in observation_signal:
            simulations.append(self._weighted_choice(observations))
        return pd.DataFrame.from_records(simulations).fillna(0)

    def _weighted_choice(self, amount: int):
        return {x: 1 for x in np.random.choice(self.sample_indices, amount, False, self.sample_weights)}


def test():
    """

    :rtype : None
    """

    import scipy.sparse as sparse

    row, col = 100, 100
    np.random.seed(77)
    df = pd.DataFrame(sparse.random(row, col, density=0.15).A).apply(np.ceil)

    df.loc[0] = [1 if x < 20 else 0 for x in range(0, df.shape[1])]
    df.loc[1] = [1 if x > 13 and x < 35 else 0 for x in range(0, df.shape[1])]
    df.loc[2] = [1 if x > 80 else 0 for x in range(0, df.shape[1])]

    m = MutEx(background=df, permutations=1000)

    pd.set_option('display.max_columns', 1000)
    print(df.loc[[0, 1, 2]])

    print("\nExample - 1 thread \n----------")

    r = m.calculate([4, 5, 6], parallel=False)
    print(r)

    print("\nExample - multi-threaded \n----------")

    r = m.calculate([0, 1, 2])
    print(r)

    random.seed(18)
    group_generator = (random.sample(df.index.tolist(), random.sample([2, 3, 4], 1)[0]) for x in range(10))
    result_list = [m.calculate(g) for g in group_generator]
    print(pd.DataFrame.from_records([r.__dict__ for r in result_list]))


if __name__ == "__main__":
    test()
