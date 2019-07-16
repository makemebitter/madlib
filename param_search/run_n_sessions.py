"""Summary

"""
import multiprocessing
# from multiprocessing import Process
import psycopg2
import argparse
import json
import time
import numpy as np
from collections import defaultdict


def get_connection():
    """Summary

    Returns:
        TYPE: Description
    """
    connection = psycopg2.connect(user="yuhaozhang",
                                  host="127.0.0.1",
                                  port="15432",
                                  database="param_search")
    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute("set optimizer = off")
    return connection, cursor


def run_query_unpack(args):
    return run_query(*args)


def run_query(worker_id, mst, mst_key, weights_table, data_table, is_summary):
    """Summary

    Args:
        worker_id (TYPE): Description
        mst (TYPE): Description
        mst_key (TYPE): Description
        weights (TYPE): Description
        weights_table (TYPE): Description
        data_table (TYPE): Description

    """

    connection, cursor = get_connection()
    # weights = "ARRAY{}".format(weights) if weights else 'NULL'
    # -- WITH prev_weights
    #                             -- AS (SELECT weights
    #                             --     FROM {}
    #                             --     WHERE mst_key='{}'
    #                             --     )
    mlp_model = "ARRAY{}".format(mst['model'])
    weights_query = """
                SELECT weights from {} where mst_key='{}'
                """.format(weights_table, mst_key)
    cursor.execute(weights_query)
    weights = cursor.fetchone()[0]
    if weights:
        weights = "ARRAY{}".format(weights)
    else:
        weights = "NULL"
    mlp_training_query = """
                            DROP TABLE IF EXISTS weights_one_partition_{};
                            CREATE TABLE weights_one_partition_{} AS (
                                SELECT madlib.mlp_minibatch_step_param_search(independent_varname,
                                    dependent_varname,
                                    {},
                                    {},
                                    {},1,1,1,NULL,
                                    {},2,1,0.5,False
                                    )::double precision[] as weights,
                                    '{}' as mst_key
                                FROM {}
                                WHERE dist_key={} 
                            )
                        """.format(worker_id,
                                   worker_id,
                                   weights,
                                   # weights_table,
                                   # mst_key,
                                   mlp_model,
                                   mst['learning_rate'],
                                   mst['lambda_value'],
                                   mst_key,
                                   data_table,
                                   # weights_table,
                                   worker_id,
                                   # mst_key
                                   )
    begin_time = time.time()
    cursor.execute(mlp_training_query)
    curr_time = time.time() - begin_time
    print("Run time for uda execution for worker {}: {}".format(worker_id, curr_time))
    loss = None
    begin_time = time.time()
    if is_summary:
        mlp_loss_query = """
                            SELECT weights[array_length(weights, 1)]
                            FROM weights_one_partition_{}
                        """.format(worker_id)
        cursor.execute(mlp_loss_query)
        record = cursor.fetchone()
        loss = record[0]
    print("partition: {}, mst: {}, loss: {}".format(worker_id, mst_key, loss))
    curr_time = time.time() - begin_time
    print("Run time for summary retrieval for worker {}: {}".format(worker_id, curr_time))

    mlp_weights_update_query = """
                    UPDATE {} SET weights = weights_one_partition_{}.weights
                    FROM  weights_one_partition_{}
                    WHERE {}.mst_key = weights_one_partition_{}.mst_key
                    """.format(weights_table,
                               worker_id,
                               worker_id,
                               weights_table,
                               worker_id)
    begin_time = time.time()
    cursor.execute(mlp_weights_update_query)
    curr_time = time.time() - begin_time
    print("Run time for weights update for worker {}: {}".format(worker_id, curr_time))
    cursor.close()
    return worker_id, mst_key, loss
    # record = cursor.fetchone()
    # print("sum of batch {} for seg id is {}\n".format(worker_id, record))
    # print("record shape is {}".format(len(record[0])))


def runInParallel(grand_schedule,
                  msts_key_map, msts, weights_table, data_table, is_summary):
    """Summary

    Args:
        grand_schedule (TYPE): Description
        msts_key_map (TYPE): Description
        msts (TYPE): Description
        weights_table (TYPE): Description
        data_table (TYPE): Description
    """
    summary = defaultdict(dict)
    pool = multiprocessing.Pool(processes=len(worker_ids))
    for i in range(len(msts_key_map)):
        args_list = []
        for worker_id in worker_ids:
            mst = grand_schedule[worker_id][i]
            key = mst_to_key(mst)
            args_list.append([worker_id,
                              mst, key, weights_table, data_table, is_summary])
        print ("arg_lists: {}".format(args_list))
        fet_res = pool.map(run_query_unpack, args_list)
        if is_summary:
            for worker_id_fetched, mst_key_fetched, loss_fetched in fet_res:
                summary[mst_key_fetched][worker_id_fetched] = loss_fetched
        print("Done with one set of mst for all workers")
    return summary


def mst_to_key(mst):
    """Summary

    Args:
        mst (TYPE): Description

    Returns:
        TYPE: Description
    """
    key_components = ["{}:{}".format(k, v) for k, v in mst.items()]
    key = '-'.join(key_components)
    return key


def rotate(l, n):
    """Summary

    Args:
        l (TYPE): Description
        n (TYPE): Description

    Returns:
        TYPE: Description
    """
    return l[-n:] + l[:-n]


def generate_schedule(worker_ids, msts):
    """Summary

    Returns:
        TYPE: Description

    Args:
        worker_ids (TYPE): Description
        msts (TYPE): Description
    """
    grand_schedule = {}
    for i, worker_id in enumerate(worker_ids):
        grand_schedule[worker_id] = rotate(msts, i)
    return grand_schedule


def create_weights_table(weights_table, msts):
    """Summary

    Args:
        weights_table (TYPE): Description
    """
    connection, cursor = get_connection()
    query = ("drop table if exists {};" +
             " create table {}" +
             " (weights double precision[], mst_key text primary key);"
             ).format(weights_table, weights_table)
    cursor.execute(query)
    for mst in msts:
        weights_insert_query = "INSERT INTO {} VALUES (NULL, '{}')".format(
            weights_table, mst_to_key(mst))
        cursor.execute(weights_insert_query)
    cursor.close()
    connection.commit()


def find_combinations(combinations, param_grid):
    """Summary

    Args:
        combinations (TYPE): Description
        param_grid (TYPE): Description
    """
    def find_combinations_helper(combinations, p, i):
        """
        :param combinations:
        :param p:
        :param i:

        Args:
            combinations (TYPE): Description
            p (TYPE): Description
            i (TYPE): Description
        """
        param_names = param_grid.keys()
        if i < len(param_names):
            for x in param_grid[param_names[i]]:
                p[param_names[i]] = x
                find_combinations_helper(combinations, p, i + 1)
        else:
            combinations.append(p.copy())
    find_combinations_helper(combinations, {}, 0)


def main(worker_ids, msts):
    """Summary

    Args:
        worker_ids (TYPE): Description
        msts (TYPE): Description
    """
    msts_key_map = {}
    # TODO think about duplicate configs
    for mst in msts:
        key = mst_to_key(mst)
        msts_key_map[key] = mst

    grand_schedule = generate_schedule(worker_ids, msts)
    print("Grand schedule: {}".format(grand_schedule))
    print("Initilizing weights table ...")
    create_weights_table(args.weights_table, msts)
    print("Initilizing weights table ... done")
    if args.is_summary:
        summary = {}
    for i in range(args.epochs):
        print ("Epoch: {}".format(i))
        epoch_summary = runInParallel(grand_schedule, msts_key_map, msts,
                                      args.weights_table, args.data_table,
                                      args.is_summary)
        if args.is_summary:
            summary[i] = epoch_summary
    if args.is_summary:
        print(summary)
        for key in msts_key_map.keys():
            learning_curve_y = []
            for i in range(args.epochs):
                partition_loss = summary[i][key]
                total_loss = np.mean(partition_loss.values())
                learning_curve_y.append(total_loss)
            print (key, learning_curve_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", nargs='?', default=5, type=int,
                        help="Number of training epochs to perform")
    parser.add_argument("-m", "--mst_config_file", nargs='?',
                        default="./msts.json",
                        help="JSON file path for mst configs")
    parser.add_argument("-w", "--weights_table", nargs='?',
                        default='n_sessions_weights', type=str,
                        help="Name of the weights table")
    parser.add_argument("-d", "--data_table", nargs='?',
                        default='iris_data_param_search', type=str,
                        help="Name of the data table")
    parser.add_argument("-s", "--is_summary", action='store_true',
                        help="Enable the flag to print out the summary")
    args = parser.parse_args()

    start_time = time.time()
    # TODO query worker number and partition info on-the-fly
    WORKER_NUMBER = 3
    worker_ids = range(WORKER_NUMBER)
    worker_ids = [w * 5 for w in worker_ids]
    with open(args.mst_config_file, "r") as read_file:
        param_grid = json.load(read_file)
    msts = []
    find_combinations(msts, param_grid)
    print("MSTS: {}".format(msts))
    main(worker_ids, msts)
    print("End to end run time: {}".format(time.time() - start_time))
