from multiprocessing import Process
import psycopg2
import itertools

def get_connection():
    connection = psycopg2.connect(user="yuhaozhang",
                                  host="127.0.0.1",
                                  port="15432",
                                  database="param_search")
    cursor = connection.cursor()
    return connection, cursor


def run_query(partition_id, config, config_key, weights):
    connection, cursor = get_connection()
    weights = "ARRAY{}".format(weights) if weights else 'NULL'
    mlp_uda_query = """
    insert into n_sessions_weights (select madlib.mlp_minibatch_step(independent_varname,dependent_varname,
    {},ARRAY[4,5,2],{},1,1,0,NULL,0.01,2,1,0.5,False)::double precision[], '{}' 
    from iris_data_param_search where dist_key={});""".format(weights, config, config_key, partition_id)

    cursor.execute(mlp_uda_query)
    cursor.close()
    connection.commit()
    # record = cursor.fetchone()
    # print("sum of batch {} for seg id is {}\n".format(partition_id, record))
    # print("record shape is {}".format(len(record[0])))


def runInParallel():
    weights_map = dict.fromkeys((range(10)))
    for i in range(len(config_map)):
        proc = []
        for worker_id in worker_ids:
            config = grand_schedule[worker_id][i]
            key = "C{}".format(config)
            weights = weights_map[key] if key in weights_map else None
            p = Process(target=run_query, args=(
                worker_id, config, key, weights))
            p.start()
            proc.append(p)
        for p in proc:
            print p.pid
            p.join()
        print("barrier here")
        for index, config in enumerate(configs):
            key = "C{}".format(config)
            weights_map[key] = query_weights(key)


def query_weights(config_id):
    _, cursor = get_connection()
    mlp_get_weights_query = "select weights from n_sessions_weights where config_key='{}'".format(
        config_id)
    cursor.execute(mlp_get_weights_query)
    record = cursor.fetchone()
    return record[0] if record else None


configs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.01]
config_map = {}
# TODO think about duplicate configs
for config in enumerate(configs):
    key = "C{}".format(config)
    config_map[key] = config
WORKER_NUMBER = 3
worker_ids = range(WORKER_NUMBER)
grand_queue = list(itertools.product(config_map.keys(), worker_ids))
grand_schedule = {}
# mst_status = {config_id:[False]*WORKER_NUMBER for config_id in config_ids}


def rotate(l, n):
    return l[-n:] + l[:-n]


for worker_id in worker_ids:
    grand_schedule[worker_id] = rotate(config_map, worker_id)


def create_weights_table():
    connection, cursor = get_connection()
    query = "drop table if exists public.n_sessions_weights; create table public.n_sessions_weights (weights double precision[], config_key text);"
    cursor.execute(query)
    cursor.close()
    connection.commit()


create_weights_table()
runInParallel()
