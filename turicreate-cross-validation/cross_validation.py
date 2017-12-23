
import json
import numpy as np
from collections import defaultdict

try:
    import turicreate as tc
except ImportError:
    tc = None


def load_nxgraph_from_sgraph(graph_path):
    sg = tc.load_sgraph(graph_path)
    import networkx as nx
    g = nx.Graph()

    # Put the nodes and edges from the SGraph into a NetworkX graph
    g.add_nodes_from(list(sg.vertices['__id']))
    g.add_edges_from([(e['__src_id'], e['__dst_id'], e['attr']) for e in sg.edges])
    return g


def save_nx_as_sgraph(df, output_path):
    sf = tc.SFrame(data=df)
    sg = tc.SGraph().add_edges(sf, src_field="source", dst_field="dest")
    sg.save(output_path)


def sgraph_to_csv(sgraph_path, output_path):
    sg = tc.load_sgraph(sgraph_path)
    sg.save(output_path, 'csv')


def sframe_to_csv(sframe_path, output_path):
    sf = tc.SFrame(sframe_path, format='array')
    sf.save(output_path, 'csv')


def sarray_to_csv(sarray_path, output_path):
    sf = tc.SArray(sarray_path)
    sf.save(output_path, 'csv')


def json_to_csv(json_path, output_path):
    with open(json_path) as json_data:
        sarray_to_csv(json.load(json_data), output_path)


def get_classification_metrics(model, targets, predictions):
    precision = tc.evaluation.precision(targets, predictions)
    accuracy = tc.evaluation.accuracy(targets, predictions)
    recall = tc.evaluation.recall(targets, predictions)
    auc = tc.evaluation.auc(targets, predictions)
    return {"recall": recall,
            "precision": precision,
            "accuracy": accuracy,
            "auc": auc
            }


def shuffle_sframe(sf, random_seed=None):
    sf["shuffle_col"] = tc.SArray.random_integers(sf.num_rows(), random_seed)
    return sf.sort("shuffle_col").remove_column("shuffle_col")


def kfold_sections(data, n_folds):
    """
    Based on scikit implementation.
    """
    Neach_section, extras = divmod(len(data), n_folds)
    section_sizes = ([0] +
                     extras * [Neach_section + 1] +
                     (n_folds - extras) * [Neach_section])
    div_points = np.array(section_sizes).cumsum()
    for i in range(n_folds):
        st = div_points[i]
        end = div_points[i + 1]
        yield st, end


def split_kfold(data, n_folds=10):
    for st, end in kfold_sections(data, n_folds):
        idx = np.zeros(len(data))
        idx[st:end] = 1
        yield data[tc.SArray(1 - idx)], data[tc.SArray(idx)]


def split_stratified_kfold(data, label='label', n_folds=10):
    if label in data.column_names():
        labels = data[label].unique()
        labeled_data = [data[data[label] == l] for l in labels]
        fold = [split_kfold(item, n_folds) for item in labeled_data]
        for _ in range(n_folds):
            train, test = tc.SFrame(), tc.SFrame()
            for f in fold:
                x_train, x_test = f.next()
                train = train.append(x_train)
                test = test.append(x_test)
            yield train, test
    else:
        yield split_kfold(data, n_folds)


def cross_validate(datasets, model_factory, model_parameters=None, evaluator=get_classification_metrics, label='label'):
    if not model_parameters:
        model_parameters = {}
    cross_val_metrics = defaultdict(list)
    for train, test in datasets:
        model = model_factory(train, **model_parameters)
        prediction = model.predict(test)
        metrics = evaluator(model, test[label], prediction)
        for k, v in metrics.iteritems():
            cross_val_metrics[k].append(v)
    return {k: np.mean(v) for k, v in cross_val_metrics.iteritems()}
