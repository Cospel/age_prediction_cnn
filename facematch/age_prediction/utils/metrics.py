from keras import backend as K


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(K.abs(cdf_true) - K.abs(cdf_pred)),  axis=-1))
    return K.mean(emd)


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 100, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 100, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae
