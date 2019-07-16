create or replace FUNCTION madlib_keras_fit_multiple_model(
    source_table VARCHAR,
    model_output_table VARCHAR,
    model_arch_table VARCHAR,
    model_selection_table VARCHAR, 
    num_iterations INTEGER,
	gpus_per_host INTEGER
) returns void as 
$$
import plpy
import time
import sys
sys.path.insert(0, "/Users/yuhaozhang/workspace/madlib/madlib/build/src/ports/greenplum/5/modules/deep_learning")
sys.path.insert(0, "/Users/yuhaozhang/workspace/madlib/madlib/build/src/ports/greenplum/5/modules")

from collections import defaultdict
import json
MINIBATCH_OUTPUT_DEPENDENT_COLNAME_DL = 'dependent_var'

from madlib_keras_helper import get_image_count_per_seg_for_minibatched_data_from_db 
from madlib_keras import get_initial_weights, compute_loss_and_metrics
from keras_model_arch_table import ModelArchSchema
from keras.models import model_from_json
import madlib_keras_serializer
import random
random.seed(42)
def rotate(l, n):
    """Summary

    Args:
        l (TYPE): Description
        n (TYPE): Description

    Returns:
        TYPE: Description
    """
    return l[-n:] + l[:-n]

def query_arch(model_arch_table, model_arch_id):
	model_arch_query = "SELECT {0}, {1} FROM {2} WHERE {3} = {4}".format(ModelArchSchema.MODEL_ARCH, 
																		 ModelArchSchema.MODEL_WEIGHTS,
																		 model_arch_table, 
																		 ModelArchSchema.MODEL_ID,
																		 model_arch_id)
	model_arch_result = plpy.execute(model_arch_query)[0]
	return model_arch_result
	
def create_model_output_table(model_output_table, model_arch_table, msts):
	weights_create_query = ("drop table if exists {};" +
						    " create table {}" +
						    " (mst_key int primary key, weights BYTEA, model_arch json);"
						   ).format(model_output_table, model_output_table)
	plpy.execute(weights_create_query)
	for mst in msts:
		model_arch_result = query_arch(model_arch_table, mst['model_arch_id'])
		model_arch = model_arch_result[ModelArchSchema.MODEL_ARCH]
		serialized_weights = get_initial_weights(model_output_table, 
												 model_arch_result,
												 False, 
												 gpus_per_host
												)
		model = model_from_json(model_arch_result[ModelArchSchema.MODEL_ARCH])
		serialized_weights = madlib_keras_serializer.serialize_state_with_nd_weights(0, model.get_weights())
		
		weights_insert_query = "INSERT INTO {} VALUES ({}, $1, $2)".format(model_output_table, mst['mst_key'])
		weights_insert_query_prepared = plpy.prepare(weights_insert_query, ["bytea", "json"])
		
		plpy.execute(weights_insert_query_prepared, [serialized_weights, json.dumps(model_arch)])
	
			 
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
	
def to_sql_str(py_str):
	return py_str.replace('\'', '\'\'')

def create_mst_schedule_table(mst_row):
	mst_temp_query = """DROP TABLE IF EXISTS mst_current_schedule; 
						CREATE TABLE mst_current_schedule (model_arch_id INTEGER, 
														   compile_params VARCHAR, 
														   fit_params VARCHAR, 
														   dist_key INTEGER, 
														   mst_key INTEGER primary key)"""
	plpy.execute(mst_temp_query)
	plpy.info("Creating temp table for msts: {}".format(mst_row))
	for mst, dist_key in zip(mst_row, dist_keys):
		mst_insert_query = "INSERT INTO mst_current_schedule VALUES ({}, '{}', '{}', {}, {})".format(mst['model_arch_id'], 
																							to_sql_str(mst['compile_params']),
																							 to_sql_str(mst['fit_params']),
																							 dist_key, 
																							 mst['mst_key'])
		plpy.execute(mst_insert_query)
		
def run_uda(model_output_table, seg_ids_train, images_per_seg_train, source_table, is_summary):
	mst_wgh_query = """DROP TABLE IF EXISTS mst_wgh; 
					   CREATE TABLE mst_wgh AS 
					   SELECT mst.*, wgh.weights, model_arch.model_arch
					   FROM mst_current_schedule mst JOIN {} wgh on mst.mst_key = wgh.mst_key
					   		JOIN {} model_arch ON mst.model_arch_id = model_arch.model_id
					   DISTRIBUTED BY (dist_key)
					   """.format(model_output_table, model_arch_table)
	plpy.execute(mst_wgh_query)

	mlp_uda_query = """
						update {} SET weights = weights_one_schedule.weights
						FROM (select madlib.fit_step_param_search(dependent_var,
																  independent_var,
																  mst_wgh.model_arch::TEXT,
																  mst_wgh.compile_params::TEXT,
																  mst_wgh.fit_params::TEXT,
																  iris.gp_segment_id,
																  ARRAY{},
																  ARRAY{},
																  0,
																  3,
																  mst_wgh.weights::BYTEA
																  )::BYTEA as weights,
							mst_wgh.mst_key as mst_key 
							from {} iris JOIN mst_wgh 
							using (dist_key)
							group by iris.dist_key, mst_wgh.mst_key
							) weights_one_schedule
						WHERE {}.mst_key = weights_one_schedule.mst_key
					""".format(model_output_table, seg_ids_train, images_per_seg_train, source_table, model_output_table)
	
									
	plpy.execute(mlp_uda_query)
	
	return 0
def query_weights(model_output_table):
	mlp_weights_query = """
								SELECT weights, mst_key FROM {}
							""".format(model_output_table)


	res = plpy.execute(mlp_weights_query)
	return res
def query_msts(model_selection_table):
	msts_query = """
				 SELECT * FROM {}
				 ORDER BY mst_key
				 """.format(model_selection_table)
	res = list(plpy.execute(msts_query))
	return res
def query_dist_keys(source_table):
	dist_key_query = """
					 SELECT DISTINCT(dist_key) FROM {}
					 ORDER BY dist_key
					 """.format(source_table)
	res = list(plpy.execute(dist_key_query))
	res = [x['dist_key'] for x in res]
	return res
	
begin_time = time.time()
# WARNING: set orca off to prevent unwanted redistribution
plpy.execute('set optimizer to off')
msts = query_msts(model_selection_table)
plpy.info(msts)
dist_keys = query_dist_keys(source_table)
plpy.info(dist_keys)
grand_schedule = generate_schedule(dist_keys, msts)
plpy.info(grand_schedule)

create_model_output_table(model_output_table, model_arch_table, msts)
seg_ids_train, images_per_seg_train = get_image_count_per_seg_for_minibatched_data_from_db(source_table)
weights_map = {}
for e in range(num_iterations):
	plpy.info("Iteration: {}".format(e))
	for i in range(len(msts)):
		mst_row = [grand_schedule[dist_key][i] for dist_key in dist_keys]
		create_mst_schedule_table(mst_row)
		run_uda(model_output_table, seg_ids_train, images_per_seg_train, source_table, True)
	res = query_weights(model_output_table)
	res_map = {x['mst_key']:x['weights'] for x in res if x['weights']}
	weights_map_one_epoch = {}
	for mst in msts:
		training_metrics = []
		training_loss = []
		model_arch = query_arch(model_arch_table, mst['model_arch_id'])[ModelArchSchema.MODEL_ARCH]
		state = res_map[mst['mst_key']]
		serialized_weights = madlib_keras_serializer.get_serialized_1d_weights_from_state(state)
		res_loss_metric = compute_loss_and_metrics(
					'madlib', source_table, "$madlib${}$madlib$".format(mst['compile_params']), model_arch,
					serialized_weights, 0, 3, seg_ids_train,
					images_per_seg_train, [], [], e)
		weights_map_one_epoch[mst['mst_key']] = res_loss_metric
	weights_map[e] = weights_map_one_epoch
plpy.info(weights_map)
plpy.info("End to end execution time: {}".format(time.time()-begin_time))
$$ language plpythonu;

-- if is_summary:
-- 	weights_map = defaultdict(dict) 
-- for e in range(epoch):
-- 	plpy.info("Epoch: {}".format(e))
-- 	if is_summary:
-- 		weights_map_one_epoch = {}
-- 	for i in range(len(msts)):
-- 		mst_row = [grand_schedule[dist_key][i] for dist_key in dist_keys]
-- 		create_mst_schedule_table(mst_row)
-- 		res = run_uda(is_summary)
-- 		if is_summary:
-- 			res_map = {x['mst_key']:x['weights'] for x in res if x['weights']}
-- 			for mst in mst_row:
-- 				mst_key = mst_to_key(mst)
-- 				weights = res_map[mst_key]
-- 				if mst == 0:
-- 					plpy.info("Initial weights: {}".format(weights))
-- 				if mst_key in weights_map_one_epoch:
-- 					weights_map_one_epoch[mst_key] += weights[-1]/3.0
-- 				else:
-- 					weights_map_one_epoch[mst_key] = weights[-1]/3.0
-- 	if is_summary:
-- 		weights_map[e] = weights_map_one_epoch
-- if is_summary:
-- 	plpy.info(weights_map)
-- # plpy.execute('DROP TABLE IF EXISTS mst_current_schedule')
SELECT madlib_keras_fit_multiple_model(
    'iris_train_packed_param_search',
    'iris_multiple_model_output_param_search',
    'model_arch_library',
    'mst_table', 
    30,
	0
)