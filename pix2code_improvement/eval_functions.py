import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from utils import clean_code
import distance
import re


def eval_cnn_model(model_instance, npz_paths, voc, words_to_include, index_value='accuracy'):
    prediction_list = []
    correct_y_list = []
    for path in tqdm(npz_paths, desc="Calculating {}".format(index_value)):
        temp = np.load(path, allow_pickle=True)
        img_data = temp['img_data']
        res = model_instance.predict({'img_data': np.expand_dims(img_data, 0)})
        code = str(temp['code'])
        counts = voc.count_words(code, words_to_count=words_to_include)

        correct_y_list.append(pd.DataFrame(counts, index=[Path(path).stem]))
        prediction_list.append(pd.DataFrame({word+"_count": res[word+"_count"][0] for word in words_to_include},
                                            index=[Path(path).stem]))

    predictions = pd.concat(prediction_list, axis=0)
    ground_truth = pd.concat(correct_y_list, axis=0)
    ratio_correct_pred = pd.DataFrame((predictions.round() == ground_truth).mean(), columns=[index_value]).T
    return ratio_correct_pred, ground_truth, predictions


def calc_code_error_ratio(ground_truth, prediction):
    return distance.levenshtein(
        clean_code(ground_truth).split(" "), clean_code(prediction).split(" "))/len(clean_code(ground_truth).split(" "))


def button_correct(str1, str2):
    return all(
        [[occurence.start() for occurence in re.finditer(button_name,str1)] == [occurence.start() for occurence in re.finditer(button_name,str2)]
         for button_name in ['btn-green','btn-orange','btn-red']])


def eval_code_error(model_instance, npz_paths, voc):
    error_list = []
    prediction_list = []
    ground_truth_list = []
    for path in tqdm(npz_paths):
        temp = np.load(path, allow_pickle=True)
        pred_code = model_instance.predict_image(temp['img_data'], voc)
        ground_truth = str(temp['code'])
        prediction_list.append(pred_code)
        ground_truth_list.append(ground_truth)
        error_list.append(calc_code_error_ratio(ground_truth, pred_code))
    res_df = pd.DataFrame({'ground_truth': ground_truth_list, 'prediction': prediction_list, 'error': error_list},
                          index=npz_paths)

    res_df['correctly_predicted'] = res_df.error == 0

    res_df['same_length'] = res_df.ground_truth.str.split(" ").apply(len) == res_df.prediction.str.split(" ").apply(len)
    res_df['active_button_correct'] = (res_df.ground_truth.str.split('close_square_bracket', 1, expand=True).iloc[:,0]
                                       == res_df.prediction.str.split('close_square_bracket', 1, expand=True).iloc[:,0])
    res_df['button_color_correct'] = res_df.apply(lambda row: button_correct(row.ground_truth, row.prediction), axis=1)

    return res_df
