def correct(pred, y):
    pred = pred.argmax(dim=1)
    return pred.eq(y)


def incorrect(pred, y):
    correct_ = correct(pred, y)
    return ~correct_


def accuracy(pred, y):
    correct_ = correct(pred, y)
    return correct_.sum().item() / len(pred)


def accuracy_all(pred, y, indices):
    pred = pred.to('cpu')
    y = y.to('cpu')
    return [accuracy(pred[idx], y[idx])
            for idx in indices]
