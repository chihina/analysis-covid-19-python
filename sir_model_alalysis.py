import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import codecs
import os

# 都道府県人口データ取得
with codecs.open("prefecture_population.csv", "r", "Shift-JIS", "ignore") as file:
    df_population = pd.read_table(file, delimiter=",", index_col=0)

# 東京都の人口に対する比率である人口比カラムを追加する
df_population["人口比"] = df_population["人口"] / df_population.loc["東京都"]["人口"]

# 都道府県名を一旦, インデックスから外す
df_population = df_population.reset_index()

# 余計な文字列を取り除く
df_population["都道府県名"] = df_population["都道府県名"].str.replace('県', '').str.replace('東京都', '東京').str.replace('府', '')

# 再び都道府県名をインデックスに設定する
df_population = df_population.set_index('都道府県名')

# ルンゲクッタ法クラスの定義(複数の地域を扱うクラス)
class MultiRungeKutta(object):
    
#     インスタンスを生成した時に動く
    def __init__(self, n, start_x, start_y):
#         分割個数 nをセット
        self.n = n
    
#     開始時間・終了時間をセット
        self.start_t = start_t
        self.finish_t = finish_t
        
#         時間刻みを計算してセット
        self.h = (finish_t - start_t) / n
    
#         時間の配列を定義する
        self.t_array = np.linspace(start_t, finish_t, num=n) 
           
#     他県の感染者データを取得する(例えば, 愛知について考えているなら愛知以外の感染者データを取得)
    def get_i_other_prefectures(self, j, t, prefecture_name):
        
#         他県の感染者データを保存するリスト
        other_prefecutures_i = []
        
#         今考えている以外の県の感染者データ取得
        for idx, prefecture_name_key in enumerate(self.init_value_dic.keys()):

#             今考えている県はパス
            if prefecture_name_key == prefecture_name:
                pass

            # 今考えていない県の感染者データ取得
            else:
                other_prefecutures_i.append(self.i_array[idx, j] * df_population.loc[prefecture_name]["人口比"])
        
#         他県の感染者データを平均化して返す
        return sum(other_prefecutures_i) / len(other_prefecutures_i) 
                
                
#     ルンゲクッタ法で用いるsを求まめるためのfを計算するメソッド(Sついて)        
    def cal_func_s(self, j, t, s, i, r, beta, gamma, prefecture_name):

        # 考えていない県以外の感染者数を考慮に入れる
        other_prefecutures_i = self.get_i_other_prefectures(j, t, prefecture_name)
        return -beta * s * i - other_prefecutures_i * df_population.loc[prefecture_name]["人口比"] * 0.1

    #     ルンゲクッタ法で用いるsを求めるためのfを計算するメソッド(Iについて)
    def cal_func_i(self, j, t, s, i, r, beta, gamma, prefecture_name):

        # 考えていない県以外の感染者数を考慮に入れる
        other_prefecutures_i = self.get_i_other_prefectures(j, t, prefecture_name)
        return beta * s * i - gamma * i + other_prefecutures_i * df_population.loc[prefecture_name]["人口比"] * 0.1

#         ルンゲクッタ法で用いるsを求めるためのfを計算するメソッド(Rについて)
    def cal_func_r(self, j, t, s, i, r, beta, gamma, prefecture_name):
        return gamma * i
    
#     ルンゲクッタ法での計算のメソッド
    def exe_cal(self, init_value_dic):
        """Returns value calculated by runge kutta algorithm

        Parameters:
        ----------
        init_value_list : list
            Init value of y or dy/dt
            
        equation_value_list : list
            Define equation value 
            
        Returns:
        ----------
        calculated value list : numpy.ndarray
            return calculated value list by runge kutta algorithm 
            
        """
                  
        #         計算した結果を保存する配列を初期化しておく(行が県名, 列が時間)
        #         県の数を取得
        prefectures_count = len(list(init_value_dic.keys()))
        self.s_array = np.zeros((prefectures_count, n))
        self.i_array = np.zeros((prefectures_count, n))
        self.r_array = np.zeros((prefectures_count, n))

        #         1つの県ごとに初期値の設定をしていく
        for idx, prefecture_name in enumerate(init_value_dic.keys()):
            
            #         初期値をセット
            self.s_array[idx, 0] = init_value_dic[prefecture_name]["S"]
            self.i_array[idx, 0] = init_value_dic[prefecture_name]["I"]
            self.r_array[idx, 0] = init_value_dic[prefecture_name]["R"]
        
#         後の計算のためそれぞれの県についてのパラメータをセットする
        self.init_value_dic = init_value_dic
    
#         ルンゲクッタ法での計算開始
        for j in range(0, self.n-1):

            # 何番目の点を計算しているかを出力する
            print("Now j:{}, finish:{}".format(j, self.n-1))

#             それぞれの県ごとに計算する
            for idx, prefecture_name in enumerate(self.init_value_dic.keys()):
#                 計算に必要となるs1, s2, s3, s4をそれぞれの変数について計算する
                
#                 時間jの時のt, S, I, Rを取得する
                t_j, s_j, i_j, r_j = self.t_array[j], self.s_array[idx, j], self.i_array[idx, j], self.r_array[idx, j]
                
#                 その県におけるパラメータ β, γを取得する
                beta, gamma = self.init_value_dic[prefecture_name]["beta"], self.init_value_dic[prefecture_name]["gamma"]
            
                s_s_1 = self.cal_func_s(j, t_j, s_j, i_j, r_j, beta, gamma, prefecture_name)
                i_s_1 = self.cal_func_i(j, t_j, s_j, i_j, r_j, beta, gamma, prefecture_name)
                r_s_1 = self.cal_func_r(j, t_j, s_j, i_j, r_j, beta, gamma, prefecture_name)

                s_s_2 = self.cal_func_s(j, t_j, s_j+self.h*s_s_1/2, i_j+self.h*i_s_1/2
                                , r_j+self.h*r_s_1/2, beta, gamma, prefecture_name)
                i_s_2 = self.cal_func_i(j, t_j, s_j+self.h*s_s_1/2, i_j+self.h*i_s_1/2
                                , r_j+self.h*r_s_1/2, beta, gamma, prefecture_name)
                r_s_2 = self.cal_func_r(j, t_j, s_j+self.h*s_s_1/2, i_j+self.h*i_s_1/2
                                , r_j+self.h*r_s_1/2, beta, gamma, prefecture_name)

                s_s_3 = self.cal_func_s(j, t_j, s_j+self.h*s_s_2/2, i_j+self.h*i_s_1/2,
                                r_j+self.h*r_s_2/2, beta, gamma, prefecture_name)
                i_s_3 = self.cal_func_i(j, t_j, s_j+self.h*s_s_2/2, i_j+self.h*i_s_1/2,
                                r_j+self.h*r_s_2/2, beta, gamma, prefecture_name)       
                r_s_3 = self.cal_func_r(j, t_j, s_j+self.h*s_s_2/2, i_j+self.h*i_s_1/2,
                                r_j+self.h*r_s_2/2, beta, gamma, prefecture_name)


                s_s_4 = self.cal_func_s(j, t_j, s_j+self.h*s_s_3, i_j+self.h*i_s_3,
                                r_j+self.h*r_s_3, beta, gamma, prefecture_name)
                i_s_4 = self.cal_func_i(j, t_j, s_j+self.h*s_s_3, i_j+self.h*i_s_3,
                                r_j+self.h*r_s_3, beta, gamma, prefecture_name)
                r_s_4 = self.cal_func_r(j, t_j, s_j+self.h*s_s_3, i_j+self.h*i_s_3,
                                r_j+self.h*r_s_3, beta, gamma, prefecture_name)

    #             計算結果を配列に保存する
                self.s_array[idx][j+1] = s_j + self.h/6*(s_s_1 + 2*s_s_2 + 2*s_s_3 + s_s_4)

                self.i_array[idx][j+1] = i_j + self.h/6*(i_s_1 + 2*i_s_2 + 2*i_s_3 + i_s_4)

                self.r_array[idx][j+1] = r_j + self.h/6*(r_s_1 + 2*r_s_2 + 2*r_s_3 + r_s_4)

            
# 分割個数を設定
n = 300

# 開始時間・終了時間を設定
start_t = 0
finish_t = 300

# 47都道府県分の初期値の設定
init_value_dic = {}

# 緯度・経度データから県名のみを取得
geo_df = pd.read_csv("prefecture_coordinate_data.csv", encoding="utf-8")

# 1つの県名について考える
for prefecture_name in geo_df["nam_ja"]:
    
    # 1つの県についての初期値を保存する辞書を定義する
    init_value_dic_prefecture = {}
    
    # 東京については以下の初期値を設定する
    if prefecture_name == "東京":
        init_value_dic_prefecture["S"] = 0.999
        init_value_dic_prefecture["I"] = 0.001
        init_value_dic_prefecture["R"] = 0.000
        init_value_dic_prefecture["beta"] = 0.2
        init_value_dic_prefecture["gamma"] = 0.01
        init_value_dic[prefecture_name] = init_value_dic_prefecture
    
    # 東京以外については, 初期値を設定する
    else:
        init_value_dic_prefecture["S"] = 1
        init_value_dic_prefecture["I"] = 0
        init_value_dic_prefecture["R"] = 0.000

        # βを人口比によって変える
        init_value_dic_prefecture["beta"] = 0.2 * df_population.loc[prefecture_name]["人口比"]
        
        init_value_dic_prefecture["gamma"] = 0.01
        init_value_dic[prefecture_name] = init_value_dic_prefecture
    
# ルンゲクッタ法クラスのインスタンスを生成
runge_kutta_ins = MultiRungeKutta(n, start_t, finish_t)

# ルンゲクッタ法による計算を実行
runge_kutta_ins.exe_cal(init_value_dic)

# グラフを保存するディレクトリの指定
save_image_dir = "prefectures_sir_images"

# グラフ保存ディレクトリがなかったら作成
if not os.path.isdir(save_image_dir):
    os.mkdir(save_image_dir)

# S, I, Rを県ごとにプロットする
for idx, prefecture_name in enumerate(runge_kutta_ins.init_value_dic.keys()):
    
    # グラフを初期化する
    plt.figure()

    # S, I. Rについてプロットする
    plt.plot(runge_kutta_ins.t_array, runge_kutta_ins.s_array[idx], color="blue", label="S")
    plt.plot(runge_kutta_ins.t_array, runge_kutta_ins.i_array[idx], color="red", label="I")
    plt.plot(runge_kutta_ins.t_array, runge_kutta_ins.r_array[idx], color="green", label="R")
    
    # ラベルをセットする
    plt.xlabel("Day")
    plt.ylabel("X,Y,Z")

    # タイトルに県名をセットする
    plt.title(prefecture_name)

    # グラフにラベルを表示する
    plt.legend()

    # 画像を保存する(ファイル名に県名を入れる)
    plt.savefig(os.path.join(save_image_dir, "S_I_R_{}.png".format(prefecture_name)))


# 計算した県名を取得
index_name = list(runge_kutta_ins.init_value_dic.keys())

# 計算した時間(day)を文字列で取得
columns_name = [str(day) for day in range(1, runge_kutta_ins.t_array.shape[0] + 1)]

# 計算した感染者データを取得
infector_data = runge_kutta_ins.i_array

# 感染者データをcsvで保存する(可視化に利用)
prefecture_infector_data = pd.DataFrame(data=infector_data, index=index_name, columns=columns_name)
prefecture_infector_data.to_csv('prefecture_infector_data.csv', encoding='utf_8_sig')