"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Evaluates performance of proposal classifiation model.

"""

METRICS_LIST = [
    "accuracy_dif",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "# splits fixed",
    "# merges created",
]


def run_evaluation(fragments_graph, proposals, accepts):
    """
    Evaluates the accuracy of predictions made by a proposal
    classication model.

    Parameters
    ----------
    fragments_graphs : FragmentsGraph
        Graph generated from fragments of a predicted segmentation.
    proposals : list[frozenset]
        Proposals classified by model.
    accepts : list[frozenset]
        Accepted proposals.

    Returns
    -------
    dict
        Dictionary that stores statistics calculated for all proposal
        predictions and separately for simple and complex proposals, as
        specified in "METRICS_LIST".

    """
    # Initializations
    stats = dict()
    simple_proposals = fragments_graph.simple_proposals()
    complex_proposals = fragments_graph.complex_proposals()

    # Evaluation
    stats["Overall"] = get_stats(fragments_graph, proposals, accepts)
    stats["Simple"] = get_stats(fragments_graph, simple_proposals, accepts)
    stats["Complex"] = get_stats(fragments_graph, complex_proposals, accepts)
    return stats


def get_stats(fragments_graph, proposals, accepts):
    """
    Computes statistics that reflect the accuracy of the predictions made by
    a proposal classication model.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph generated from fragments of a predicted segmentation.
    proposals : list[frozenset]
        List of proposals of a specified "proposal_type".
    accepts : numpy.ndarray
        Accepted proposals.

    Returns
    -------
    dict
        Results of evaluation where the keys are identical to "METRICS_LIST".

    """
    n_pos = len([e for e in proposals if e in fragments_graph.gt_accepts])
    a_baseline = n_pos / (len(proposals) if len(proposals) > 0 else 1)
    tp, fp, a, p, r, f1 = get_accuracy(fragments_graph, proposals, accepts)
    stats = {
        "# splits fixed": tp,
        "# merges created": fp,
        "accuracy_dif": a - a_baseline,
        "accuracy": a,
        "precision": p,
        "recall": r,
        "f1": f1,
    }
    return stats


def get_accuracy(fragments_graph, proposals, accepts):
    """
    Computes the following metrics for a given set of predicted proposals:
    (1) true positives, (2) false positive, (3) accuracy, (4) precision,
    (5) recall, and (6) f1-score.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph generated from fragments of a predicted segmentation.
    proposals : set[frozenset]
        List of proposals of a specified "proposal_type".
    accepts : list
        Accepted proposals.

    Returns
    -------
    float, float, float, float, float, float
        Number true positives, number of false positives, accuracy, precision,
        recall, and F1-score.

    """
    tp, tn, fp, fn = get_detection_cnts(fragments_graph, proposals, accepts)
    a = (tp + tn) / len(proposals) if len(proposals) else 1
    p = 1 if tp + fp == 0 else tp / (tp + fp)
    r = 1 if tp + fn == 0 else tp / (tp + fn)
    f1 = (2 * r * p) / max(r + p, 1e-3)
    return tp, fp, a, p, r, f1


def get_detection_cnts(fragments_graph, proposals, accepts):
    """
    Computes the following values: (1) true positives, (2) true negatives,
    (3) false positive, and (4) false negatives.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph generated from fragments of a predicted segmentation.
    proposals : set[frozenset]
        List of proposals of a specified "proposal_type".
    accepts : list
        Accepted proposals.

    Returns
    -------
    float, float, float, float
        Number of true positives, true negatives, false positives, and false
        negatives.

    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for p in proposals:
        if p in fragments_graph.gt_accepts:
            tp += 1 if p in accepts else 0
            fn += 1 if p not in accepts else 0
        else:
            fp += 1 if p in accepts else 0
            tn += 1 if p not in accepts else 0
    return tp, tn, fp, fn
