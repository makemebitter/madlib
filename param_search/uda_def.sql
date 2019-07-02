create or replace FUNCTION mlp_param_search() returns void as 
$$
import plpy

msts = [0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00001]
dist_keys = [0, 5, 10]

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
	weights_create_query = ("drop table if exists public.{};" +
             " create table public.{}" +
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

grand_schedule = generate_schedule(dist_keys, msts)
weights_table = "uda_weights"
create_weights_table(weights_table, msts)

plpy.info(grand_schedule)

def create_mst_schedule_table(mst_row):
	mst_temp_query = "DROP TABLE IF EXISTS mst_current_schedule; CREATE TABLE mst_current_schedule (mst_value text, dist_key int, mst_key text primary key)"
	plpy.execute(mst_temp_query)
	plpy.info("Creating temp table for msts: {}".format(mst_row))
	for mst, dist_key in zip(mst_row, dist_keys):
		mst_insert_query = "INSERT INTO mst_current_schedule VALUES ({}, {}, '{}')".format(mst, dist_key, mst_to_key(mst))
		plpy.execute(mst_insert_query)
		
def run_uda(weights):
	mst_wgh_query = "drop table if exists mst_wgh; create table mst_wgh as select * from mst_current_schedule mst JOIN uda_weights wgh using(mst_key) distributed by (dist_key)"
	plpy.execute(mst_wgh_query)
	mlp_uda_query = """
						update {} SET weights = weights_one_schedule.weights
						FROM (select madlib.mlp_minibatch_step_param_search(independent_varname,
																			dependent_varname, 
																			mst_wgh.weights,
																			ARRAY[4,5,2],
																			mst_wgh.mst_value::float,
																			1,1,0,NULL,0.0001,2,1,0.5,False
																			)::double precision[] as weights,
							mst_wgh.mst_key as mst_key 
							from iris_data_param_search iris JOIN mst_wgh 
							using (dist_key)
							group by iris.dist_key, mst_wgh.mst_key
							) weights_one_schedule
						WHERE {}.mst_key = weights_one_schedule.mst_key
					""".format(weights_table, weights_table, weights_table)

									
	plpy.execute(mlp_uda_query)
	
def query_weights(weights_table, mst_key):
	mlp_get_weights_query = ("select weights" +
                             " from {}" +
                             " where mst_key='{}'"
                             ).format(weights_table, mst_key)
	plpy.info(mlp_get_weights_query)
	res = plpy.execute(mlp_get_weights_query)
	return res[0]["weights"]

# WARNING: set orca off to prevent unwanted redistribution
plpy.execute('set optimizer to off')
	
for i in range(len(msts)):
	mst_row = [grand_schedule[dist_key][i] for dist_key in dist_keys]
	create_mst_schedule_table(mst_row)
	run_uda(None)
# plpy.execute('DROP TABLE IF EXISTS mst_current_schedule')
$$ language plpythonu;

SELECT mlp_param_search();