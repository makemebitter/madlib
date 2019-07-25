DROP TABLE IF EXISTS iris_data_dl;
CREATE TABLE iris_data_dl(
    id serial,
    attributes numeric[],
    class_text varchar
);
INSERT INTO iris_data_dl(id, attributes, class_text) VALUES
(1,ARRAY[5.1,3.5,1.4,0.2],'Iris-setosa'),
(2,ARRAY[4.9,3.0,1.4,0.2],'Iris-setosa'),
(3,ARRAY[4.7,3.2,1.3,0.2],'Iris-setosa'),
(4,ARRAY[4.6,3.1,1.5,0.2],'Iris-setosa'),
(5,ARRAY[5.0,3.6,1.4,0.2],'Iris-setosa'),
(6,ARRAY[5.4,3.9,1.7,0.4],'Iris-setosa'),
(7,ARRAY[4.6,3.4,1.4,0.3],'Iris-setosa'),
(8,ARRAY[5.0,3.4,1.5,0.2],'Iris-setosa'),
(9,ARRAY[4.4,2.9,1.4,0.2],'Iris-setosa'),
(10,ARRAY[4.9,3.1,1.5,0.1],'Iris-setosa'),
(11,ARRAY[5.4,3.7,1.5,0.2],'Iris-setosa'),
(12,ARRAY[4.8,3.4,1.6,0.2],'Iris-setosa'),
(13,ARRAY[4.8,3.0,1.4,0.1],'Iris-setosa'),
(14,ARRAY[4.3,3.0,1.1,0.1],'Iris-setosa'),
(15,ARRAY[5.8,4.0,1.2,0.2],'Iris-setosa'),
(16,ARRAY[5.7,4.4,1.5,0.4],'Iris-setosa'),
(17,ARRAY[5.4,3.9,1.3,0.4],'Iris-setosa'),
(18,ARRAY[5.1,3.5,1.4,0.3],'Iris-setosa'),
(19,ARRAY[5.7,3.8,1.7,0.3],'Iris-setosa'),
(20,ARRAY[5.1,3.8,1.5,0.3],'Iris-setosa'),
(21,ARRAY[5.4,3.4,1.7,0.2],'Iris-setosa'),
(22,ARRAY[5.1,3.7,1.5,0.4],'Iris-setosa'),
(23,ARRAY[4.6,3.6,1.0,0.2],'Iris-setosa'),
(24,ARRAY[5.1,3.3,1.7,0.5],'Iris-setosa'),
(25,ARRAY[4.8,3.4,1.9,0.2],'Iris-setosa'),
(26,ARRAY[5.0,3.0,1.6,0.2],'Iris-setosa'),
(27,ARRAY[5.0,3.4,1.6,0.4],'Iris-setosa'),
(28,ARRAY[5.2,3.5,1.5,0.2],'Iris-setosa'),
(29,ARRAY[5.2,3.4,1.4,0.2],'Iris-setosa'),
(30,ARRAY[4.7,3.2,1.6,0.2],'Iris-setosa'),
(31,ARRAY[4.8,3.1,1.6,0.2],'Iris-setosa'),
(32,ARRAY[5.4,3.4,1.5,0.4],'Iris-setosa'),
(33,ARRAY[5.2,4.1,1.5,0.1],'Iris-setosa'),
(34,ARRAY[5.5,4.2,1.4,0.2],'Iris-setosa'),
(35,ARRAY[4.9,3.1,1.5,0.1],'Iris-setosa'),
(36,ARRAY[5.0,3.2,1.2,0.2],'Iris-setosa'),
(37,ARRAY[5.5,3.5,1.3,0.2],'Iris-setosa'),
(38,ARRAY[4.9,3.1,1.5,0.1],'Iris-setosa'),
(39,ARRAY[4.4,3.0,1.3,0.2],'Iris-setosa'),
(40,ARRAY[5.1,3.4,1.5,0.2],'Iris-setosa'),
(41,ARRAY[5.0,3.5,1.3,0.3],'Iris-setosa'),
(42,ARRAY[4.5,2.3,1.3,0.3],'Iris-setosa'),
(43,ARRAY[4.4,3.2,1.3,0.2],'Iris-setosa'),
(44,ARRAY[5.0,3.5,1.6,0.6],'Iris-setosa'),
(45,ARRAY[5.1,3.8,1.9,0.4],'Iris-setosa'),
(46,ARRAY[4.8,3.0,1.4,0.3],'Iris-setosa'),
(47,ARRAY[5.1,3.8,1.6,0.2],'Iris-setosa'),
(48,ARRAY[4.6,3.2,1.4,0.2],'Iris-setosa'),
(49,ARRAY[5.3,3.7,1.5,0.2],'Iris-setosa'),
(50,ARRAY[5.0,3.3,1.4,0.2],'Iris-setosa'),
(51,ARRAY[7.0,3.2,4.7,1.4],'Iris-versicolor'),
(52,ARRAY[6.4,3.2,4.5,1.5],'Iris-versicolor'),
(53,ARRAY[6.9,3.1,4.9,1.5],'Iris-versicolor'),
(54,ARRAY[5.5,2.3,4.0,1.3],'Iris-versicolor'),
(55,ARRAY[6.5,2.8,4.6,1.5],'Iris-versicolor'),
(56,ARRAY[5.7,2.8,4.5,1.3],'Iris-versicolor'),
(57,ARRAY[6.3,3.3,4.7,1.6],'Iris-versicolor'),
(58,ARRAY[4.9,2.4,3.3,1.0],'Iris-versicolor'),
(59,ARRAY[6.6,2.9,4.6,1.3],'Iris-versicolor'),
(60,ARRAY[5.2,2.7,3.9,1.4],'Iris-versicolor'),
(61,ARRAY[5.0,2.0,3.5,1.0],'Iris-versicolor'),
(62,ARRAY[5.9,3.0,4.2,1.5],'Iris-versicolor'),
(63,ARRAY[6.0,2.2,4.0,1.0],'Iris-versicolor'),
(64,ARRAY[6.1,2.9,4.7,1.4],'Iris-versicolor'),
(65,ARRAY[5.6,2.9,3.6,1.3],'Iris-versicolor'),
(66,ARRAY[6.7,3.1,4.4,1.4],'Iris-versicolor'),
(67,ARRAY[5.6,3.0,4.5,1.5],'Iris-versicolor'),
(68,ARRAY[5.8,2.7,4.1,1.0],'Iris-versicolor'),
(69,ARRAY[6.2,2.2,4.5,1.5],'Iris-versicolor'),
(70,ARRAY[5.6,2.5,3.9,1.1],'Iris-versicolor'),
(71,ARRAY[5.9,3.2,4.8,1.8],'Iris-versicolor'),
(72,ARRAY[6.1,2.8,4.0,1.3],'Iris-versicolor'),
(73,ARRAY[6.3,2.5,4.9,1.5],'Iris-versicolor'),
(74,ARRAY[6.1,2.8,4.7,1.2],'Iris-versicolor'),
(75,ARRAY[6.4,2.9,4.3,1.3],'Iris-versicolor'),
(76,ARRAY[6.6,3.0,4.4,1.4],'Iris-versicolor'),
(77,ARRAY[6.8,2.8,4.8,1.4],'Iris-versicolor'),
(78,ARRAY[6.7,3.0,5.0,1.7],'Iris-versicolor'),
(79,ARRAY[6.0,2.9,4.5,1.5],'Iris-versicolor'),
(80,ARRAY[5.7,2.6,3.5,1.0],'Iris-versicolor'),
(81,ARRAY[5.5,2.4,3.8,1.1],'Iris-versicolor'),
(82,ARRAY[5.5,2.4,3.7,1.0],'Iris-versicolor'),
(83,ARRAY[5.8,2.7,3.9,1.2],'Iris-versicolor'),
(84,ARRAY[6.0,2.7,5.1,1.6],'Iris-versicolor'),
(85,ARRAY[5.4,3.0,4.5,1.5],'Iris-versicolor'),
(86,ARRAY[6.0,3.4,4.5,1.6],'Iris-versicolor'),
(87,ARRAY[6.7,3.1,4.7,1.5],'Iris-versicolor'),
(88,ARRAY[6.3,2.3,4.4,1.3],'Iris-versicolor'),
(89,ARRAY[5.6,3.0,4.1,1.3],'Iris-versicolor'),
(90,ARRAY[5.5,2.5,4.0,1.3],'Iris-versicolor'),
(91,ARRAY[5.5,2.6,4.4,1.2],'Iris-versicolor'),
(92,ARRAY[6.1,3.0,4.6,1.4],'Iris-versicolor'),
(93,ARRAY[5.8,2.6,4.0,1.2],'Iris-versicolor'),
(94,ARRAY[5.0,2.3,3.3,1.0],'Iris-versicolor'),
(95,ARRAY[5.6,2.7,4.2,1.3],'Iris-versicolor'),
(96,ARRAY[5.7,3.0,4.2,1.2],'Iris-versicolor'),
(97,ARRAY[5.7,2.9,4.2,1.3],'Iris-versicolor'),
(98,ARRAY[6.2,2.9,4.3,1.3],'Iris-versicolor'),
(99,ARRAY[5.1,2.5,3.0,1.1],'Iris-versicolor'),
(100,ARRAY[5.7,2.8,4.1,1.3],'Iris-versicolor'),
(101,ARRAY[6.3,3.3,6.0,2.5],'Iris-virginica'),
(102,ARRAY[5.8,2.7,5.1,1.9],'Iris-virginica'),
(103,ARRAY[7.1,3.0,5.9,2.1],'Iris-virginica'),
(104,ARRAY[6.3,2.9,5.6,1.8],'Iris-virginica'),
(105,ARRAY[6.5,3.0,5.8,2.2],'Iris-virginica'),
(106,ARRAY[7.6,3.0,6.6,2.1],'Iris-virginica'),
(107,ARRAY[4.9,2.5,4.5,1.7],'Iris-virginica'),
(108,ARRAY[7.3,2.9,6.3,1.8],'Iris-virginica'),
(109,ARRAY[6.7,2.5,5.8,1.8],'Iris-virginica'),
(110,ARRAY[7.2,3.6,6.1,2.5],'Iris-virginica'),
(111,ARRAY[6.5,3.2,5.1,2.0],'Iris-virginica'),
(112,ARRAY[6.4,2.7,5.3,1.9],'Iris-virginica'),
(113,ARRAY[6.8,3.0,5.5,2.1],'Iris-virginica'),
(114,ARRAY[5.7,2.5,5.0,2.0],'Iris-virginica'),
(115,ARRAY[5.8,2.8,5.1,2.4],'Iris-virginica'),
(116,ARRAY[6.4,3.2,5.3,2.3],'Iris-virginica'),
(117,ARRAY[6.5,3.0,5.5,1.8],'Iris-virginica'),
(118,ARRAY[7.7,3.8,6.7,2.2],'Iris-virginica'),
(119,ARRAY[7.7,2.6,6.9,2.3],'Iris-virginica'),
(120,ARRAY[6.0,2.2,5.0,1.5],'Iris-virginica'),
(121,ARRAY[6.9,3.2,5.7,2.3],'Iris-virginica'),
(122,ARRAY[5.6,2.8,4.9,2.0],'Iris-virginica'),
(123,ARRAY[7.7,2.8,6.7,2.0],'Iris-virginica'),
(124,ARRAY[6.3,2.7,4.9,1.8],'Iris-virginica'),
(125,ARRAY[6.7,3.3,5.7,2.1],'Iris-virginica'),
(126,ARRAY[7.2,3.2,6.0,1.8],'Iris-virginica'),
(127,ARRAY[6.2,2.8,4.8,1.8],'Iris-virginica'),
(128,ARRAY[6.1,3.0,4.9,1.8],'Iris-virginica'),
(129,ARRAY[6.4,2.8,5.6,2.1],'Iris-virginica'),
(130,ARRAY[7.2,3.0,5.8,1.6],'Iris-virginica'),
(131,ARRAY[7.4,2.8,6.1,1.9],'Iris-virginica'),
(132,ARRAY[7.9,3.8,6.4,2.0],'Iris-virginica'),
(133,ARRAY[6.4,2.8,5.6,2.2],'Iris-virginica'),
(134,ARRAY[6.3,2.8,5.1,1.5],'Iris-virginica'),
(135,ARRAY[6.1,2.6,5.6,1.4],'Iris-virginica'),
(136,ARRAY[7.7,3.0,6.1,2.3],'Iris-virginica'),
(137,ARRAY[6.3,3.4,5.6,2.4],'Iris-virginica'),
(138,ARRAY[6.4,3.1,5.5,1.8],'Iris-virginica'),
(139,ARRAY[6.0,3.0,4.8,1.8],'Iris-virginica'),
(140,ARRAY[6.9,3.1,5.4,2.1],'Iris-virginica'),
(141,ARRAY[6.7,3.1,5.6,2.4],'Iris-virginica'),
(142,ARRAY[6.9,3.1,5.1,2.3],'Iris-virginica'),
(143,ARRAY[5.8,2.7,5.1,1.9],'Iris-virginica'),
(144,ARRAY[6.8,3.2,5.9,2.3],'Iris-virginica'),
(145,ARRAY[6.7,3.3,5.7,2.5],'Iris-virginica'),
(146,ARRAY[6.7,3.0,5.2,2.3],'Iris-virginica'),
(147,ARRAY[6.3,2.5,5.0,1.9],'Iris-virginica'),
(148,ARRAY[6.5,3.0,5.2,2.0],'Iris-virginica'),
(149,ARRAY[6.2,3.4,5.4,2.3],'Iris-virginica'),
(150,ARRAY[5.9,3.0,5.1,1.8],'Iris-virginica');

DROP TABLE IF EXISTS iris_train, iris_test;
-- Set seed so results are reproducible
SELECT setseed(0);
SELECT madlib.train_test_split('iris_data_dl',     -- Source table
                               'iris',          -- Output table root name
                                0.8,            -- Train proportion
                                NULL,           -- Test proportion (0.2)
                                NULL,           -- Strata definition
                                NULL,           -- Output all columns
                                NULL,           -- Sample without replacement
                                TRUE            -- Separate output tables
                              );
SELECT COUNT(*) FROM iris_train;

DROP TABLE IF EXISTS mlp_prediction_dl;
DROP TABLE IF EXISTS iris_train_packed, iris_train_packed_summary, iris_train_packed_param_search_summary, iris_train_packed_param_search;
SELECT madlib.training_preprocessor_dl('iris_train',         -- Source table
                                       'iris_train_packed',  -- Output table
                                       'class_text',         -- Dependent variable
                                       'attributes',         -- Independent variable
6
                                        );
										
CREATE TABLE iris_train_packed_param_search as (select *, mod(row_number() over(), 3)*5  as dist_key from iris_train_packed) distributed by (dist_key);

CREATE TABLE iris_train_packed_param_search_summary as SELECT * FROM iris_train_packed_summary;

DROP TABLE IF EXISTS iris_test_packed, iris_test_packed_summary, iris_test_packed_param_search;
SELECT madlib.validation_preprocessor_dl('iris_test',          -- Source table
                                         'iris_test_packed',   -- Output table
                                         'class_text',         -- Dependent variable
                                         'attributes',         -- Independent variable
                                         'iris_train_packed'   -- From training preprocessor step
                                          );
CREATE TABLE iris_test_packed_param_search as (select *, mod(row_number() over(), 3)*5  as dist_key from iris_test_packed) distributed by (dist_key);
SELECT * FROM iris_test_packed_summary;

DROP TABLE IF EXISTS model_arch_library;
SELECT madlib.load_keras_model('model_arch_library',  -- Output table,
$$
{"class_name": "Sequential", "keras_version": "2.1.6", "config": [{"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_1", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "relu", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "batch_input_shape": [null, 4], "use_bias": true, "activity_regularizer": null}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_2", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "relu", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "use_bias": true, "activity_regularizer": null}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_3", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "softmax", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 3, "use_bias": true, "activity_regularizer": null}}], "backend": "tensorflow"}
$$
::json,  -- JSON blob
                               NULL,                  -- Weights
                               'Sophie',              -- Name
                               'A simple model'       -- Descr
);


