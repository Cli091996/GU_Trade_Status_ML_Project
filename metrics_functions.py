from matplotlib import pyplot as plt 

def threshold_viz(metrics, model_name='Model'):
    # Extract metrics
    thresholds = metrics['thresholds']
    precision_list = metrics['precision_list']
    recall_list = metrics['recall_list']
    accuracy_list = metrics['accuracy_list']

    # Calculate differences
    precision_diffs = [j - i for i, j in zip(precision_list[:-1], precision_list[1:])]
    recall_diffs = [j - i for i, j in zip(recall_list[:-1], recall_list[1:])]
    accuracy_diffs = [j - i for i, j in zip(accuracy_list[:-1], accuracy_list[1:])]
    
    # Find indices of the highest jump and biggest drop-off
    max_jump_precision_idx = precision_diffs.index(max(precision_diffs)) + 1
    max_jump_accuracy_idx = accuracy_diffs.index(max(accuracy_diffs)) + 1
    max_dropoff_recall_idx = recall_diffs.index(min(recall_diffs)) + 1

    # Calculate F1 scores
    f1_list = [2 * (p * r) / (p + r) for p, r in zip(precision_list, recall_list)]
    max_f1_idx = f1_list.index(max(f1_list))
    
    # Find all indices of the highest precision, recall, F1 score, and accuracy
    max_precision_indices = [i for i, v in enumerate(precision_list) if v == max(precision_list)]
    max_recall_indices = [i for i, v in enumerate(recall_list) if v == max(recall_list)]
    max_accuracy_indices = [i for i, v in enumerate(accuracy_list) if v == max(accuracy_list)]
    max_f1_indices = [i for i, v in enumerate(f1_list) if v == max(f1_list)]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(thresholds, precision_list, label='Precision',color='cornflowerblue',marker='.')
    ax.plot(thresholds, recall_list, label='Recall',color='dimgrey',marker='.')
    ax.plot(thresholds, accuracy_list, label='Accuracy',color='mediumseagreen',marker='.')
    ax.plot(thresholds, f1_list, label='F1 Score',color='darkviolet',marker='.')
    
    # Highlight points of interest
    ax.scatter(thresholds[max_jump_precision_idx], precision_list[max_jump_precision_idx], color='cornflowerblue', zorder=5, label='Max Jump Precision')
    ax.scatter(thresholds[max_jump_accuracy_idx], accuracy_list[max_jump_accuracy_idx], color='mediumseagreen', zorder=5, label='Max Jump Accuracy')
    ax.scatter(thresholds[max_dropoff_recall_idx], recall_list[max_dropoff_recall_idx], color='dimgrey', zorder=5, label='Max Dropoff Recall')
    
    # Highlight and annotate all the highest precision, recall, and accuracy
    for idx in max_f1_indices:
        ax.scatter(thresholds[idx], f1_list[idx], color='darkviolet', zorder=5)
    ax.annotate('Max F Score', (thresholds[max_f1_indices[0]], f1_list[max_f1_indices[0]]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='darkviolet')

    for idx in max_precision_indices:
        ax.scatter(thresholds[idx], precision_list[idx], color='cornflowerblue', zorder=5)
    ax.annotate('Max Precision', (thresholds[max_precision_indices[0]], precision_list[max_precision_indices[0]]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='cornflowerblue')
    
    for idx in max_recall_indices:
        ax.scatter(thresholds[idx], recall_list[idx], color='dimgrey', zorder=5)
    ax.annotate('Max Recall', (thresholds[max_recall_indices[0]], recall_list[max_recall_indices[0]]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='dimgrey')
    
    for idx in max_accuracy_indices:
        ax.scatter(thresholds[idx], accuracy_list[idx], color='mediumseagreen', zorder=5)
    ax.annotate('Max Accuracy', (thresholds[max_accuracy_indices[0]], accuracy_list[max_accuracy_indices[0]]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='mediumseagreen')
    
    # Annotate other points of interest
    ax.annotate('Max Jump In Precision', (thresholds[max_jump_precision_idx], precision_list[max_jump_precision_idx]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='cornflowerblue')
    ax.annotate('Max Jump In Accuracy', (thresholds[max_jump_accuracy_idx], accuracy_list[max_jump_accuracy_idx]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='mediumseagreen')
    ax.annotate('Max Dropoff In Recall', (thresholds[max_dropoff_recall_idx], recall_list[max_dropoff_recall_idx]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='dimgrey')
    
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{model_name}: Precision, Recall, and Accuracy vs. Decision Threshold')

    # Customize grid
    ax.grid(color='gray', linestyle='--', linewidth=0.25)
    
    # Add legends outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Adjust the layout to make room for the legend
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Show the plot
    return plt.show()

def custom_classification_metrics(true_arrray, predictions_array):

    metrics_dict = {
        'thresholds':[i / 100 for i in range(50, 100, 5)],
        'precision_list' : [],
        'recall_list' : [],
        'accuracy_list': [],
    }
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for threshold in metrics_dict['thresholds']:
        for i in range(len(true_arrray)):
            a = predictions_array[i]
            b = true_arrray.iloc[i]
            if a > threshold:
                if b == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if b == 1:
                    FN += 1
                else:
                    TN += 1
        
        if TP + FP > 0:
            precision = round(TP / (TP + FP),3)
            metrics_dict['precision_list'].append(precision)
        else:
            precision = 0
            metrics_dict['precision_list'].append(precision)
        
        if TP + FN > 0:
            recall = TP / (TP + FN)
            metrics_dict['recall_list'].append(recall)
        else:
            recall = 0
            metrics_dict['recall_list'].append(recall)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        metrics_dict['accuracy_list'].append(accuracy)
    return metrics_dict