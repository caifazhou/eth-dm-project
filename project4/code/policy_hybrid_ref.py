#!/usr/bin/env python2.7

import numpy as np
import numpy.random
from numpy.linalg import inv

# sigma = 0.3
alpha = 1.35
# alpha = 1 + np.sqrt(np.log(2 / sigma) / 2)
Dim_user = 6    # dimension of user features
Dim_arti = 6    # dimension of article features

# set the input data
M = dict()
b = dict()
w = dict()
n = dict()

# set the parameters
ttt = 1
z_t = 0
Artikel = {}
last_article_id = "Nah"

num_f = Dim_user
num_a = Dim_arti

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(articles):
    global Artikel, M, b, M_0, b_0, beta, n, w

    Artikel = articles

    M_0 = np.identity(num_f*num_a)
    b_0 = np.zeros(num_f*num_a)

    beta = np.matrix(np.zeros(num_f*num_a)).transpose()

    for article_id in Artikel:
        x = str(article_id)
        M[x] = np.identity(num_f)
        b[x] = np.zeros(num_f)
        w[x] = np.matrix(np.zeros(num_f)).transpose()
        n[x] = 1


# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global ttt, M, b, M_0, b_0, beta, n, w ,phy
    if reward == -1:
        return
    y_t = reward

    x = str(last_article_id)

    phy = (np.dot(np.matrix(Artikel[last_article_id]).transpose(), np.matrix(z_t))).flatten()

    M_0 = np.add(M_0, np.dot(phy.transpose(), phy))
    b_0 = np.add(b_0, np.multiply(phy, np.subtract(y_t, np.dot(w[x].transpose(), z_t))))

    M[x] = np.add(M[x], np.dot(np.matrix(z_t).transpose(), np.matrix(z_t)))
    b[x] = np.add(b[x], np.multiply(z_t, np.subtract(y_t, np.dot(phy, beta))))

    beta = np.dot(inv(M_0), np.matrix(b_0).transpose())
    w[x] = np.dot(inv(M[x]), np.matrix(b[x]).transpose())

    n[x] += 1
    ttt += 1


# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    best_article_id = None
    best_ucb = -1

    global ttt, z_t, M, b, last_article_id, w ,phy

    z_t = np.array(user_features)

    Zaehler = 0
    best_article_id = 0
    for article_id in articles:
        Zaehler += 1
        x = str(article_id)

        phy = (np.dot(np.matrix(Artikel[article_id]).transpose(), np.matrix(z_t))).flatten()

        UCB2 = 1000 * (200 - n[x])

        if UCB2 > 0:
            UCB = UCB2
        else:
            UCB = np.dot(w[x].transpose(), z_t)+np.dot(phy, beta)+(5+ttt/225000.0)/float(n[x])

        if Zaehler == 1 or UCB > best_ucb:
            best_ucb = UCB
            best_article_id = article_id

    last_article_id = best_article_id

    return best_article_id
