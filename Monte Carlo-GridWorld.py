"""
深度强化学习课程作业

蒙特卡洛算法，直接运行此单个文件即可，不依赖于任何调用文件。
gridworld 12*12 gridworld细节设置请见结果图片
算法参考的cs234课程所讲，代码未参考任何其他来源。
输出包括policy和轨迹图片两个部分
"""

import matplotlib.pyplot as plt
import random
import numpy as np
# 用于展示任务开始时的grid world长什么样子，方便确认解是不是最优
def plot_world(world, stone_list, start_position, final_position, result=None):
    plt.figure(1)

    plt.ylim([0, len(world)-1])
    plt.xlim([0, len(world)-1])
    plt.xticks([i for i in range(len(world))], [str(i) for i in range(len(world))])
    plt.yticks([i for i in range(len(world))], [str(i) for i in range(len(world))])
    plt.grid()
    plt.title("grid world")
    plt.scatter(start_position[0], start_position[1], s=150, color="red", marker="s")
    plt.scatter(final_position[0], final_position[1], s=150, color="blue", marker="o")
    for eve in stone_list:
        plt.scatter(eve[0], eve[1], s=150, color="green", marker="^")
    if result != None:
        for i in range(len(result)-1):
            plt.plot([result[i][0], result[i+1][0]], [result[i][1], result[i+1][1]], color="red", marker="*")
        plt.savefig("mc-grid-result.png", dpi=600)
    else:
        plt.savefig("grid.png", dpi=600)
        plt.show()

# 根据动作和当前状态，决定下一时刻的状态, max_trick为最大坐标值
def action_result(action, current_state, max_trick):
    if action == "up":
        if current_state[1] == max_trick:
            return current_state
        else:
            return (current_state[0], current_state[1]+1)
    elif action == "down":
        if current_state[1] == 0:
            return current_state
        else:
            return (current_state[0], current_state[1]-1)
    elif action == "left":
        if current_state[0] == 0:
            return current_state
        else:
            return (current_state[0]-1, current_state[1])
    elif action == "right":
        if current_state[0] == max_trick:
            return current_state
        else:
            return (current_state[0]+1, current_state[1])
    else:
        raise IOError


# 奖励函数的指定，十分重要！！！
def get_reward(state, final_position, stone_list, current_state):
    if state == current_state:
        return -7
    if state == final_position:
        return 30
    elif state in stone_list:
        return -30
    else:
        return -1

# 获得最大的q值
def get_maxq(qtable, state):
    temp = []
    for i in range(len(qtable)):
        temp.append(qtable[i][state[0]][state[1]])
    maxone = max(temp)
    argmax = np.argmax(temp)
    return maxone, argmax


# 往后看不是往前看
# g_reward = lambda tmp, gama, t: sum([(gama**(i-t))*tmp[2][i] for i in range(t, len(tmp[2]))])

def g_reward(tmp, gama, t):
    res = 0
    for i in range(t, len(tmp[2])):
        res += (gama**(i-t))*tmp[2][i]
    return res


def print_policy(policy_, stone_list, final_position):
    with open('mc-policy.txt', "w", encoding="utf-8") as f:
        for x in range(len(policy_)):
            for y in range(len(policy_[x])):
                if (x, y) in stone_list:
                    print("({},{}):{}".format(x, y, "障碍物"), end="; ", file=f)
                    print("({},{}):{}".format(x, y, "障碍物"), end="; ")
                elif (x, y) == final_position:
                    print("({},{}):{}".format(x, y, "终点"), end="; ", file=f)
                    print("({},{}):{}".format(x, y, "终点"), end="; ")
                else:
                    print("({},{}):{}".format(x, y, action[policy_[x][y]]), end="; ", file=f)
                    print("({},{}):{}".format(x, y, action[policy_[x][y]]), end="; ")
            print("", file=f)
            print("")

if __name__ == "__main__":
    # random.seed(1)
    # 生成12*12大小的二维网格世界，且world[i][j] = (i, j)
    world = [[(i,j) for j in range(12)] for i in range(12)]
    # 设置障碍物
    stone_list = [(6,4), (10,8), (1,2), (2,3), (5,6), (10,9), (1,8), (3,9),(8,7),(3,4), (2,6),(7,6)]
    # 设置入口与出口
    start_position = (1, 1)
    final_position = (6, 7)
    # plot_world(world, stone_list, start_position, final_position)
    # 动作
    action = ["up", "down", "left", "right"]


    # 蒙特卡洛 算法解决grid world问题
    # q table用于存储动作状态值 三维列表，初始值均为0
    q_table = [[[0 for j in range(len(world))] for i in range(len(world))] for k in range(4)]
    # n table用于存储访问次数 三维列表，初始值均为0
    n_table = [[[0 for j in range(len(world))] for i in range(len(world))] for k in range(4)]
    policy = [[0 for j in range(len(world))] for i in range(len(world))]
    episodes = 300
    gama = 0.9
    # epsilon = 0.4
    # 先进行蒙特卡罗仿真，并记录s a r 序列
    print("开始蒙特卡洛仿真，结果不确定性较高，若长时间没有结果请重新运行")
    for episode in range(episodes):
        # epsilon = 1 / (episode + 1)
        epsilon = 0.3
        # print(episode)
        current_state = start_position  # 仿真的起点
        save = [current_state]
        temp = [[],[],[]]  # 存储s a r
        # 更新epsilon
        # epsilon = 1 / (episode + 1)
        # 获取序列
        while True:
            # 动作的选择
            if random.randint(1, 100) / 100 > epsilon:
                action_index = policy[current_state[0]][current_state[1]]
            else:
                action_index = random.randint(0, 3)
            # print(action_index)
            next_state = action_result(action[action_index], current_state, 11)
            # print(next_state)
            reward = get_reward(next_state, final_position, stone_list, current_state)
            # 存储结果
            temp[0].append(current_state)
            temp[1].append(action_index)
            temp[2].append(reward)
            # 时间步长
            current_state = next_state
            save.append(current_state)
            # 终止条件
            if reward == 30 or reward == -30:  # 回合结束条件
                # temp[0].append(next_state)  # 注意状态序列要多一个
                # print(save)
                # print(temp)
                break

        # print("cut111")
        # 更新
        visited = [[[0 for j in range(len(world))] for i in range(len(world))] for k in range(4)]
        # Gt = 0
        for t in range(len(temp[2])):
            if visited[temp[1][t]][temp[0][t][0]][temp[0][t][1]] == 0: # 代表在episode中首次访问s a
                visited[temp[1][t]][temp[0][t][0]][temp[0][t][1]] = 1 # 代表访问过了
                n_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]] += 1
                # Gt = gama * (temp[2][t] + Gt)
                Gt = g_reward(temp, gama, t)
                q_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]] += \
                    (1/n_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]])*(Gt-q_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]])
                # q_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]] += 0.01*(Gt-q_table[temp[1][t]][temp[0][t][0]][temp[0][t][1]])
                # 更新策略
                _, policy[temp[0][t][0]][temp[0][t][1]] = get_maxq(q_table, temp[0][t])

        # # 更新策略-由于Q值更新完毕，因此需要将所有episode经历过的
        # for update_state in temp[0]:
        #     _, policy[update_state[0]][update_state[1]] = get_maxq(q_table, update_state)

    # 进行推理,policy是最后我们要得到的东西
    begin = start_position
    state = begin
    res = [state]
    print("begin:", state, end=";")
    for i in range(30):

        a_index = policy[state[0]][state[1]]
        next_state = action_result(action[a_index], state, 11)
        print(next_state, end=";")
        res.append(next_state)
        if next_state == final_position:
            print("bingo!")
            print("共走了",i+1,"步")
            print()
            plot_world(world, stone_list, begin, final_position, res)
            print("使用q-learning训练并推理产生的结果图见根目录'mc-grid-result.png'")
            print("使用q-learning训练并推理产生的策略已保存在'mc-policy.txt'")
            print("mc策略：")
            print_policy(policy, stone_list, final_position)
            break
        if i== 29:
            print("\n算法未收敛，请重新设置参数！")
        state = next_state












