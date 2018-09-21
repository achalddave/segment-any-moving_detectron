import numpy as np


def get_iou_index(coco_eval, thr):
    # Modified from json_dataset_evaluator._log_detection_eval_metrics
    ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                    (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
    iou_thr = coco_eval.params.iouThrs[ind]
    assert np.isclose(iou_thr, thr)
    return ind


def accumulate_with_more_info(coco_eval, p=None):
    '''
    Modified from cocoeval.py.

    Differences:
        * Store recalls at every score level.

    Accumulate per image evaluation results and store the result in
    coco_eval.eval

    :param p: input params for evaluation
    :return: eval object (like in coco_eval.eval)
    '''
    import datetime
    import time
    if not coco_eval.evalImgs:
        print('Please run evaluate() first')
        return
    # allows input customized parameters
    if p is None:
        p = coco_eval.params
    p.catIds = p.catIds if p.useCats == 1 else [-1]

    T           = len(p.iouThrs)
    R           = len(p.recThrs)
    K           = len(p.catIds) if p.useCats else 1
    A           = len(p.areaRng)
    M           = len(p.maxDets)
    precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    recall      = -np.ones((T,R,K,A,M))
    scores      = -np.ones((T,R,K,A,M))

    # create dictionary for future indexing
    _pe = coco_eval._paramsEval
    catIds = _pe.catIds if _pe.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, _pe.areaRng))
    setM = set(_pe.maxDets)
    setI = set(_pe.imgIds)

    all_precisions = -np.ones((T, len(setI), K, A, M))
    all_recalls = -np.ones((T, len(setI), K, A, M))

    # get inds to evaluate
    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(_pe.imgIds)
    A0 = len(_pe.areaRng)
    # retrieve E at each category, area range, and max number of detections
    for k, k0 in enumerate(k_list):
        Nk = k0*A0*I0
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                E = [coco_eval.evalImgs[Nk + Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg==0 )
                if npig == 0:
                    continue
                tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))
                    r = np.zeros((R,))

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                            r[ri] = rc[pi]
                    except:
                        pass
                    precision[t,:,k,a,m] = np.array(q)
                    scores[t,:,k,a,m] = np.array(ss)
                    recall[t,:,k,a,m] = np.array(r)
    return {
        'params': p,
        'counts': [T, R, K, A, M],
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision': precision,
        'recall':   recall,
        'all_precisions': all_precisions,
        'all_recalls': all_recalls,
        'scores': scores,
    }
