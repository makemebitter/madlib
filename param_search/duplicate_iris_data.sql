DROP TABLE IF EXISTS iris_data_param_search_1000;
CREATE TABLE iris_data_param_search_1000 AS (SELECT iris.dependent_varname, iris.independent_varname, iris.row_number() over() as dist_key
                                                FROM iris_data_param_search iris, 
                                                    (SELECT a AS i 
                                                     FROM generate_series(0, 999) AS s(a) 
                                                     ORDER BY(i)
                                                    ) id
                                            )
                                            DISTRIBUTED BY (dist_key);
                                            