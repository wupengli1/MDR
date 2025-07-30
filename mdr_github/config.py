
class Config(object):
    r""" Model Configuration. """
    output_dir = './Model'
    output_dir1 = './Model_1'
    output_dir2 = './Model_2'
    data_path = ['/../Data/Person_Dialog_EN', '/../Data/Person_Dialog_ZH']
    model_metric = True
    consis_model_dir_EN = '../Consis_Model/consis_model_EN.ckpt'
    consis_model_dir_ZH = '../Consis_Model/consis_model_ZH.ckpt'
    # Data language
    data_language = 'EN'
    model_file = 'None'
    max_len = 128
    # Word vector dimension
    embedding_dim = 768
    # ---------------Inference phase
    # Generate sentences
    generate_max_len = 32
    repetition_penalty =100
    topk = 0
    topp = 0.9
    # The temperature of the similarity
    temp = 0.5
    # To optimize the parameters
    batch_size = 16
    accumulate_grad_batches = 1
    epochs = 5
    lr = 1e-4  # The initial vector
    seed = 2021
    print_per_step = 500//batch_size
    result_path =  './result'
