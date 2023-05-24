# 删除相关学习模型、训练和测试数据以及各种log
# 以便重新训练、重头分析，同时清理空间
# python删除文件的方法 os.remove(path) path指的是文件的绝对路径,如：
# os.remove(r"E:\code\practice\data\1.py")  #删除文件
# os.rmdir(r"E:\code\practice\data\2")      #删除文件夹（只能删除空文件夹）
# path_data = "E:\code\practice\data"#

import os


def del_file(path_data):
    for ld in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, ld)  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


path_list = [
    r".\logs",
    r".\train_output",
    r".\test_output",
    r".\models\PER_DDQN",
    r".\analysis"
]

for path in path_list:
    del_file(path)
