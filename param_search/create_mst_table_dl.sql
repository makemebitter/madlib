create or replace FUNCTION madlib_keras_create_mst_table_from_combination(
    model_selection_table VARCHAR, 
	model_arch_id_list INTEGER[],
    compile_params_list TEXT[],
    fit_params_list TEXT[]
) returns void as 
$$
from collections import OrderedDict
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
def create_mst_table(model_selection_table):
	create_query = """
					DROP TABLE IF EXISTS {};
					CREATE TABLE {} (mst_key SERIAL, 
						model_arch_id INTEGER, 
						compile_params VARCHAR, 
						fit_params VARCHAR, 
						unique (model_arch_id, compile_params, fit_params)
					   );
				   """.format(model_selection_table, model_selection_table)
	plpy.execute(create_query)

def insert_into_mst_table(model_selection_table, msts):
	for mst in msts:
		model_arch_id = mst['model_arch_id']
		compile_params = mst['compile_params']
		fit_params = mst['fit_params']
		insert_query = """
						INSERT INTO {}(model_arch_id, 
					  				   compile_params, 
					  		           fit_params) 
					  	VALUES ({}, 
							    '{}', 
							    '{}')
					   """.format(model_selection_table, model_arch_id, compile_params.replace('\'', '\'\''), fit_params.replace('\'', '\'\''))
		plpy.execute(insert_query)
msts = []
param_grid = OrderedDict([('model_arch_id' , model_arch_id_list), ('compile_params' , compile_params_list), ('fit_params', fit_params_list)])
find_combinations(msts, param_grid)
plpy.info(msts)
create_mst_table(model_selection_table)
insert_into_mst_table(model_selection_table, msts)
$$ language plpythonu;
SELECT madlib_keras_create_mst_table_from_combination(
    'mst_table',
	ARRAY[1],
    ARRAY['loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.1)'', metrics=[''accuracy'']',
		  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.01)'', metrics=[''accuracy'']',
		  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.001)'', metrics=[''accuracy'']'
		 ],
	ARRAY['batch_size=5, epochs=1',
		  'batch_size=10, epochs=1'
		 ]
);
SELECT * FROM mst_table;











-- OR do it manually
-- INSERT INTO mst_table(model_arch_id, 
-- 					  compile_params, 
-- 					  fit_params) 
-- 					  VALUES (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.1)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1'),
-- 							  (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.01)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1'),
-- 							  (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.001)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1'),
-- 							  (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.0005)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1'),
-- 							  (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.0001)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1'),
-- 							  (1, 
-- 							  'loss=''categorical_crossentropy'', optimizer=''Adam(lr=0.00001)'', metrics=[''accuracy'']', 
-- 							  'batch_size=16, epochs=1')