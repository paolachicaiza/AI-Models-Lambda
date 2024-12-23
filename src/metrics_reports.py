
def classes_name(map_encode_data_output_classes, type_predict):
    target_names_classes = []
    for classes in map_encode_data_output_classes[type_predict]:
        target_name_class = map_encode_data_output_classes[type_predict][classes]
        target_names_classes.append(target_name_class)
    return target_names_classes


def classes_report(report_model, classes_names_report):
    performance = []
    for classes in classes_names_report:
        key_performance_metric = ['documentID', 'precision']
        value_performance_metric = []
        precision_class = report_model[classes]['precision']
        value_performance_metric.append(classes)
        value_performance_metric.append(precision_class)
        classes_report_metrics = dict(zip(key_performance_metric, value_performance_metric))
        performance.append(classes_report_metrics)
    return performance
