# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:42:09 2019

@author: hcji
"""

import keras.backend as K

def pearson(y_pred, y_true):
    x = K.flatten(y_true)
    y = K.flatten(y_pred)  
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def loss(y_pred, y_true):
    x = K.flatten(y_true)
    y = K.flatten(y_pred)  
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return -K.mean(r)