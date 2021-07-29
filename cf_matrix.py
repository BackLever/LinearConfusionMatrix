import numpy as np
import matplotlib.pyplot as plt

def linear_confusion_matrix(y_true, y_pred, number=10, annot=False,
                            cmap='Blues', text_colors=["black", "white"], linewidth=2,
                            min_value=None, max_value=None, fontsize=None):
    
    if min_value is None:
        min_value = np.min(y_true) >= np.min(y_pred) and np.min(y_pred) or np.min(y_true)
    
    if max_value is None:
        max_value = np.max(y_true) >= np.max(y_pred) and np.max(y_true) or np.max(y_pred)
    
    cf_matrix = np.zeros((number, number))
    
    data = np.array([y_true[:, 0], y_pred[:, 0]]).T
    data = data[data[:, 0].argsort()]
    value_label = np.linspace(min_value, max_value, number + 1)
    
    row_sum = np.zeros(number)
    for i in range(number):
        data_tmp = data[(data[:, 0] >= value_label[i]) & (data[:, 0] < value_label[i+1])]
        for j in range(number):
            if j == number - 1:
                data_in_range = data_tmp[(data_tmp[:, 1] >= value_label[j])
                                         & (data_tmp[:, 1] <= value_label[j+1])]
            else:
                data_in_range = data_tmp[(data_tmp[:, 1] >= value_label[j])
                                         & (data_tmp[:, 1] < value_label[j+1])]
            cf_matrix[i, j] = len(data_in_range)
            row_sum[i] = row_sum[i] + cf_matrix[i, j]
    
    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(cf_matrix, cmap=cmap)

    threshold_c = im.norm(np.max(cf_matrix))/1.6
    threshold_a = im.norm(np.max(cf_matrix))/10
    
    if fontsize is None:
        fontsize = min(180 / number, 25)
    ax.set_xticks(np.arange(-.5, number, 1))
    ax.set_yticks(np.arange(-.5, number, 1))
    ax.set_xticklabels(np.around(value_label, 3), fontsize=fontsize)
    ax.set_yticklabels(np.around(value_label, 3), fontsize=fontsize)
    ax.grid(color='w', linestyle='-', linewidth=linewidth)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel("Predict", fontsize=fontsize)
    ax.set_ylabel("Actual", fontsize=fontsize)
    
    if annot:
        for i in range(number):
            for j in range(number):
                string = int(cf_matrix[i, j])
                if (int(im.norm(cf_matrix[i, j]) > threshold_a)):
                    string = "{}\n({})".format(string, np.around(cf_matrix[i, j] / row_sum[i], 2))
                text = ax.text(j, i, string, ha="center", va="center", size=fontsize,
                               color=text_colors[int(im.norm(cf_matrix[i, j]) > threshold_c)])
                
    plt.show()
    
    correct_sum = 0
    for i in range(number):
        correct_sum = correct_sum + cf_matrix[i, i]
    print("Accuracy: {}".format(np.around(correct_sum/len(data), 2)))