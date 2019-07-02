"""Summary

"""
from multiprocessing import Process
import psycopg2
import argparse
import json
import time

def get_connection():
    """Summary

    Returns:
        TYPE: Description
    """
    connection = psycopg2.connect(user="yuhaozhang",
                                  host="127.0.0.1",
                                  port="15432",
                                  database="param_search")
    cursor = connection.cursor()
    cursor.execute("set optimizer = off")
    connection.commit()
    return connection, cursor


def run_query(partition_id, mst, mst_key, weights, weights_table, data_table):
    """Summary

    Args:
        partition_id (TYPE): Description
        mst (TYPE): Description
        mst_key (TYPE): Description
        weights (TYPE): Description
        weights_table (TYPE): Description
        data_table (TYPE): Description

    """

    connection, cursor = get_connection()
    weights = "ARRAY{}".format(weights) if weights else 'NULL'
    mlp_model = "ARRAY{}".format(mst['model'])
    mlp_uda_query = """
    insert into {} (select madlib.mlp_minibatch_step(independent_varname,dependent_varname,
    {},{},{},1,1,0,NULL,{},2,1,0.5,False)::double precision[], '{}' 
    from {} where dist_key={});""".format(weights_table, weights, mlp_model, mst['learning_rate'], mst['lambda_value'], mst_key, data_table, partition_id)

    cursor.execute(mlp_uda_query)
    cursor.close()
    connection.commit()
    # record = cursor.fetchone()
    # print("sum of batch {} for seg id is {}\n".format(partition_id, record))
    # print("record shape is {}".format(len(record[0])))


def runInParallel(grand_schedule, msts_key_map, msts, weights_table, data_table):
    """Summary

    Args:
        grand_schedule (TYPE): Description
        msts_key_map (TYPE): Description
        msts (TYPE): Description
        weights_table (TYPE): Description
        data_table (TYPE): Description
    """
    weights_map = dict.fromkeys((range(10)))
    for i in range(len(msts_key_map)):
        proc = []
        for worker_id in worker_ids:
            mst = grand_schedule[worker_id][i]
            key = mst_to_key(mst)
            weights = weights_map[key] if key in weights_map else None
            # TODO Don't mix worker_id and partition_id
            p = Process(target=run_query, args=(
                worker_id, mst, key, weights, weights_table, data_table))
            p.start()
            print("Received mst:{} on worker {} with pid:{}".format(
                mst_to_key(mst), worker_id, p.pid))
            proc.append(p)
        for p in proc:
            p.join()
        print("Done with one set of mst for all workers")
        for mst in msts:
            key = mst_to_key(mst)
            weights_map[key] = query_weights(key, weights_table)


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


def query_weights(mst_key, weights_table):
    """Summary

    Args:
        mst_key (TYPE): Description
        weights_table (TYPE): Description

    Returns:
        TYPE: Description

    """
    _, cursor = get_connection()
    mlp_get_weights_query = ("select weights" +
                             " from {}" +
                             " where mst_key='{}'"
                             ).format(weights_table, mst_key)
    cursor.execute(mlp_get_weights_query)
    record = cursor.fetchone()
    return record[0] if record else None


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
    for worker_id in worker_ids:
        grand_schedule[worker_id] = rotate(msts, worker_id)
    return grand_schedule


def create_weights_table(weights_table):
    """Summary

    Args:
        weights_table (TYPE): Description
    """
    connection, cursor = get_connection()
    query = ("drop table if exists public.{};" +
             " create table public.{}" +
             " (weights double precision[], mst_key text);"
             ).format(weights_table, weights_table)
    cursor.execute(query)
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
    create_weights_table(args.weights_table)
    for i in range(args.epochs):
        print ("Epoch: {}".format(i))
        runInParallel(grand_schedule, msts_key_map, msts,
                      args.weights_table, args.data_table)

#TODO
# 1. fix worker ids
# 2. update/insert 

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
    parser.add_argument("-t", "--data_table", nargs='?',
                        default='iris_data_param_search', type=str,
                        help="Name of the data table")
    args = parser.parse_args()

    start_time = time.time()
    # TODO query worker number and partition info on-the-fly
    WORKER_NUMBER = 3
    worker_ids = range(WORKER_NUMBER)
    with open(args.mst_config_file, "r") as read_file:
        param_grid = json.load(read_file)
    msts = []
    find_combinations(msts, param_grid)
    print("MSTS: {}".format(msts))
    main(worker_ids, msts)
    print("End to end runtime: {}".format(time.time() - start_time))
