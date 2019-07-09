create or replace FUNCTION mlp_param_search(data_table text, weights_table text, epoch int, is_summary boolean) returns void as 
$$
import plpy
import time
from collections import defaultdict

def mst_to_key(mst_value):
	return "C{}".format(mst_value)
	
def rotate(l, n):
    """Summary

    Args:
        l (TYPE): Description
        n (TYPE): Description

    Returns:
        TYPE: Description
    """
    return l[-n:] + l[:-n]
	
def create_weights_table(weights_table, msts):
	weights_create_query = ("drop table if exists {};" +
             " create table {}" +
             " (weights double precision[], mst_key text primary key);"
             ).format(weights_table, weights_table)
	plpy.execute(weights_create_query)
	for mst in msts:
		weights_insert_query = "INSERT INTO {} VALUES (NULL, '{}')".format(weights_table, mst_to_key(mst))
		plpy.execute(weights_insert_query)
	
			 
def generate_schedule(dist_keys, msts):
    """Summary

    Returns:
        TYPE: Description

    Args:
        dist_keys (TYPE): Description
        msts (TYPE): Description
    """
    grand_schedule = {}
    for index, dist_key in enumerate(dist_keys):
        grand_schedule[dist_key] = rotate(msts, index)
    return grand_schedule


def create_mst_schedule_table(mst_row):
	mst_temp_query = "DROP TABLE IF EXISTS mst_current_schedule; CREATE TABLE mst_current_schedule (mst_value text, dist_key int, mst_key text primary key)"
	plpy.execute(mst_temp_query)
	plpy.info("Creating temp table for msts: {}".format(mst_row))
	for mst, dist_key in zip(mst_row, dist_keys):
		mst_insert_query = "INSERT INTO mst_current_schedule VALUES ({}, {}, '{}')".format(mst, dist_key, mst_to_key(mst))
		plpy.execute(mst_insert_query)
		
def run_uda(is_summary):
	mst_wgh_query = "drop table if exists mst_wgh; create table mst_wgh as select * from mst_current_schedule mst JOIN uda_weights wgh using(mst_key) distributed by (dist_key)"
	plpy.execute(mst_wgh_query)

	mlp_uda_query = """
						update {} SET weights = weights_one_schedule.weights
						FROM (select madlib.mlp_minibatch_step_param_search(independent_varname,
																			dependent_varname, 
																			mst_wgh.weights,
																			ARRAY[4,200,100,2],
																			mst_wgh.mst_value::float,
																			1,1,1,NULL,0.0001,2,1,0.5,False
																			)::double precision[] as weights,
							mst_wgh.mst_key as mst_key 
							from {} iris JOIN mst_wgh 
							using (dist_key)
							group by iris.dist_key, mst_wgh.mst_key
							) weights_one_schedule
						WHERE {}.mst_key = weights_one_schedule.mst_key
					""".format(weights_table, data_table, weights_table)
	
									
	plpy.execute(mlp_uda_query)
	mlp_weights_query = """
							SELECT weights, mst_key FROM {}
						""".format(weights_table)
	
	res = plpy.execute(mlp_weights_query) if is_summary else None
	return res
begin_time = time.time()
# WARNING: set orca off to prevent unwanted redistribution
plpy.execute('set optimizer to off')
# Learning rate of 0 for fetching the initial weights
msts = [0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00001]
dist_keys = [0, 5, 10]
grand_schedule = generate_schedule(dist_keys, msts)
create_weights_table(weights_table, msts)
plpy.info(grand_schedule)
if is_summary:
	weights_map = defaultdict(dict) 
for e in range(epoch):
	plpy.info("Epoch: {}".format(e))
	if is_summary:
		weights_map_one_epoch = {}
	for i in range(len(msts)):
		mst_row = [grand_schedule[dist_key][i] for dist_key in dist_keys]
		create_mst_schedule_table(mst_row)
		res = run_uda(is_summary)
		if is_summary:
			res_map = {x['mst_key']:x['weights'] for x in res if x['weights']}
			for mst in mst_row:
				mst_key = mst_to_key(mst)
				weights = res_map[mst_key]
				if mst == 0:
					plpy.info("Initial weights: {}".format(weights))
				if mst_key in weights_map_one_epoch:
					weights_map_one_epoch[mst_key] += weights[-1]/3.0
				else:
					weights_map_one_epoch[mst_key] = weights[-1]/3.0
	if is_summary:
		weights_map[e] = weights_map_one_epoch
if is_summary:
	plpy.info(weights_map)
# plpy.execute('DROP TABLE IF EXISTS mst_current_schedule')
plpy.info("End to end execution time: {}".format(time.time()-begin_time))
$$ language plpythonu;

SELECT mlp_param_search('iris_data_param_search_1000', 'uda_weights', 10, true);