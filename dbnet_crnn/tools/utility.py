import os
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def parse_args():
    params = dict()

    # prediction engine
    params["r_optim"] = True
    params["use_tensorrt"] = False
    params["gpu_mem"] = 8000

    # text detector
    params["det_algorithm"] = 'DB'
    params["det_max_side_len"] = 1500

    # DB Net
    params["det_db_thresh"] = 0.3
    params["det_db_box_thresh"] = 0.5
    params["det_db_unclip_ratio"] = 2.0

    # text recognizer
    params["rec_algorithm"] = 'CRNN'
    params["rec_image_shape"] = "3, 32, 320"
    params["rec_char_type"] = 'ch'
    params["rec_batch_num"] = 30
    params["max_text_length"] = 25
    params["rec_char_dict_path"] = "dbnet_crnn/ppocr/utils/keys.txt"
    params["use_space_char"] = True
    params["enable_mkldnn"] = False
    params["use_zero_copy_run"] = False
    return params


def create_predictor(args, mode, model_path):
    model_dir = model_path
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"
    assert os.path.exists(model_file_path)
    assert os.path.exists(params_file_path)
    config = AnalysisConfig(model_file_path, params_file_path)

    # use CPU
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(6)
    if args['enable_mkldnn']:
        config.enable_mkldnn()

    # config.enable_memory_optim()
    config.disable_glog_info()

    if args['use_zero_copy_run']:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
        config.switch_use_feed_fetch_ops(True)

    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_tensor(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors
