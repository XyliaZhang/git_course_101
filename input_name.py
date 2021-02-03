import pandas as pd
#import numpy as np
#import re
from sklearn.preprocessing import OneHotEncoder
#import time
from sklearn import decomposition
#from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
#import jieba
from sklearn import cluster,  metrics


def find_same_word(input_s, g_codes_list, db_path1):
    df1 = pd.read_csv(db_path1, sep=",", header=None,dtype="str", encoding= 'utf-8-sig')
    g_codes_list = g_codes_list.replace("'", "").replace("\"", "").replace(" ","").strip("[]").split(",")

    df_name = df1.iloc[:, 1]
    df_gcode = df1.iloc[:, 2]
    # 先找出同類別的商標們
    g_list = []
    n = 0
    a = len(df_gcode)
    for x in range(len(g_codes_list)):
        for i in range(a):
            try:
                if str(g_codes_list[x]) == str(df_gcode[i]):
                    if df_name[i] not in g_list:
                        g_list.append(str(df_name[i]))
            except:
                pass
    # #有一樣字的就選出來
    # #把問題也加入 list
    output_list = [input_s]

    for j in str(input_s):
        b = len(g_list)
        for i in range(b):
            c = len(str(g_list[i]))
            for k in range(c):
                if str(j) == str(g_list[i])[k]:
                    if str(g_list[i]) not in output_list:
                        output_list.append(str(g_list[i]))
    new_df = pd.DataFrame(output_list, columns=["output"])
    #     new_df.to_csv("./data/input_temp.csv",index=False)

    # 2. one hot encoding 製作每個商標的數列標籤
    df = new_df['output']
    out_put_one_list = []
    for i in range(len(df)):
        for j in df[i]:
            if j not in out_put_one_list:
                out_put_one_list.append(j)
    out_put_one_df = pd.DataFrame(out_put_one_list)
    dummy_df = pd.get_dummies(out_put_one_df)
    n_df = pd.DataFrame()
    for i in range(len(df)):
        m_df = pd.DataFrame()
        for j in df[i]:
            for k in dummy_df.columns:
                if str(j) == str(k[2]):
                    m_df = m_df.append(dummy_df[k].T)
        n_df = n_df.append(pd.DataFrame(m_df.sum(axis=0)).T)
    array_train_one_sk = n_df.to_numpy()
    # print(array_train_one_sk)
    df_train_one_sk = n_df

    # 3.PCA降維
    X_pca = array_train_one_sk
    pca = decomposition.PCA(n_components=3)
    X_pca_done = pca.fit_transform(X_pca)

    X_pca_df = pd.DataFrame(X_pca_done)

    # 4. 開始各種分群
    model_a1 = AgglomerativeClustering(n_clusters=30, linkage='average')
    c_a1 = model_a1.fit_predict(X_pca_done)
    label_a1 = pd.Series(model_a1.labels_)

    model_a2 = AgglomerativeClustering(n_clusters=30, linkage='complete')
    c_a2 = model_a2.fit_predict(X_pca_done)
    label_a2 = pd.Series(model_a2.labels_)

    model_a3 = AgglomerativeClustering(n_clusters=30, linkage='ward')
    c_a3 = model_a3.fit_predict(X_pca_done)
    label_a3 = pd.Series(model_a3.labels_)

    model_a4 = AgglomerativeClustering(n_clusters=30, linkage='single')
    c_a4 = model_a4.fit_predict(X_pca_done)
    label_a4 = pd.Series(model_a4.labels_)

    model_k1 = KMeans(n_clusters=30, init="random")
    c_k1 = model_k1.fit_predict(X_pca_done)
    label_k1 = pd.Series(model_k1.labels_)

    model_k2 = KMeans(n_clusters=30, init="k-means++")
    c_k2 = model_k2.fit_predict(X_pca_done)
    label_k2 = pd.Series(model_k2.labels_)

    clus_df = pd.DataFrame()
    clus_df["ag_average"] = c_a1
    clus_df["ag_complete"] = c_a2
    clus_df["ag_ward"] = c_a3
    clus_df["ag_single"] = c_a4
    clus_df["kmeans"] = c_k1
    clus_df["kmeans_plus"] = c_k2

    df_new = pd.concat([df, clus_df], axis=1)
    # df_new.to_csv("./data/trained.csv", encoding="utf-8-sig")

    # 5. 和問題同群的有
    q_a_list = []
    for i in df_new.columns:
        q_a_list.append(df_new[i][0])
    all_list = []

    for j in range(1, len(q_a_list)):
        one_list = []
        one_no_list = []
        for k in range(1, len(df_new)):
            if str(df_new.iloc[:, j][k]) == str(q_a_list[j]):
                one_list.append(df_new.iloc[:, 0][k])
                one_no_list.append(df_new.iloc[:, -1][k])
        all_list.append(one_list)

    all_df = pd.DataFrame(all_list)

    # 6. 分群結果計算分數
    silhouette_a1 = metrics.silhouette_score(X_pca_done, label_a1)
    silhouette_a2 = metrics.silhouette_score(X_pca_done, label_a2)
    silhouette_a3 = metrics.silhouette_score(X_pca_done, label_a3)
    silhouette_a4 = metrics.silhouette_score(X_pca_done, label_a4)
    silhouette_k1 = metrics.silhouette_score(X_pca_done, c_k1)
    silhouette_k2 = metrics.silhouette_score(X_pca_done, c_k2)
    silhouette_score_list = [silhouette_a1, silhouette_a2, silhouette_a3, silhouette_a4, silhouette_k1, silhouette_k2]
    n = len(silhouette_score_list)
    silhouette_percentage = []
    for i in silhouette_score_list:
        silhouette_percentage.append(float("{:.2f}".format((i / n) * 100)))
    q_a_list = []
    for i in df_new.columns:
        q_a_list.append(df_new[i][0])
    unique_list = []
    for j in range(1, len(q_a_list)):
        for k in range(1, len(df_new)):
            if str(df_new.iloc[:, j][k]) == str(q_a_list[j]):
                if df_new.iloc[:, 0][k] not in unique_list:
                    if df_new.iloc[:, 0][k] != None:
                        unique_list.append(df_new.iloc[:, 0][k])
    score_list = [0] * len(unique_list)
    for i in range(len(unique_list)):
        if unique_list[i] in all_list[0]:
            score_list[i] += silhouette_percentage[0]
        if unique_list[i] in all_list[1]:
            score_list[i] += silhouette_percentage[1]
        if unique_list[i] in all_list[2]:
            score_list[i] += silhouette_percentage[2]
        if unique_list[i] in all_list[3]:
            score_list[i] += silhouette_percentage[3]
        if unique_list[i] in all_list[4]:
            score_list[i] += silhouette_percentage[4]
        if unique_list[i] in all_list[5]:
            score_list[i] += silhouette_percentage[5]

        score_list[i] = "{:.2f}".format(score_list[i])

    df_score = pd.DataFrame()
    df_score["name"] = unique_list
    df_score["score"] = score_list
    result_df = df_score.sort_values(by="score", ascending=False).head(10)

    rank_name = result_df.iloc[:, 0].tolist()
    #print(type(rank_name))
    rank_score = result_df.iloc[:, 1].tolist()
    return rank_name,rank_score


#if __name__ == "__main__":
