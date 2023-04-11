def evaluate_indicators(tp: int, fp: int, tn: int, fn: int):
    DSC = 2*tp/(fp+2*tp+fn)
    mIOU = ((tp/(tp+fp+fn))+(tn/(tn+fn+fp)))/2
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    OA = (tp+tn)/(tp+tn+fp+fn)

    return DSC, mIOU, recall, precision, f1, OA
