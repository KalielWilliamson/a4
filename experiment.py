import multiprocessing
import os

from src import pendulum_policy_iteration, \
    pendulum_q_learning, \
    pendulum_value_iteration, \
    frozen_lake_policy_iteration, \
    frozen_lake_q_learning, \
    frozen_lake_value_iteration

if __name__ == '__main__':
    # delete artifacts directory

    # create artifacts folder with subfolders
    os.mkdir('artifacts')
    os.mkdir('artifacts/pendulum_policy_iteration')
    os.mkdir('artifacts/pendulum_q_learning')
    os.mkdir('artifacts/pendulum_value_iteration')
    os.mkdir('artifacts/frozen_lake_policy_iteration')
    os.mkdir('artifacts/frozen_lake_q_learning')
    os.mkdir('artifacts/frozen_lake_value_iteration')

    # use multiprocessing to run experiments
    num_cpu = multiprocessing.cpu_count()

    # running all experiments in parallel
    pool = multiprocessing.Pool(processes=num_cpu)
    pool.apply_async(pendulum_policy_iteration.run)
    pool.apply_async(frozen_lake_policy_iteration.main())

    pool.apply_async(pendulum_q_learning.run)
    pool.apply_async(frozen_lake_q_learning.run)

    pool.apply_async(pendulum_value_iteration.run)
    pool.apply_async(frozen_lake_value_iteration.run)

    pool.close()
    pool.join()









