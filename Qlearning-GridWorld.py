"""
深度强化学习课程作业

q-learning算法，直接运行此单个文件即可，不依赖于任何调用文件。
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
        plt.savefig("qlearning-grid-result.png", dpi=600)
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
        return -3
    if state == final_position:
        return 10
    elif state in stone_list:
        return -10
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

def print_policy(policy_, stone_list, final_position):
    with open('qlearning-policy.txt', "w", encoding="utf-8") as f:
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
    # 生成12*12大小的二维网格世界，且world[i][j] = (i, j)
    world = [[(i,j) for j in range(12)] for i in range(12)]
    # 设置11个障碍物
    stone_list = [(6,4), (10,8), (1,2), (2,3), (5,6), (10,9), (1,8), (3,9), (9,5), (10,7), (9,2), (9,7), (6, 9),(8,7),(8,8),(8,9)]
    # 设置入口与出口
    start_position = (1, 1)
    final_position = (9, 8)
    # plot_world(world, stone_list, start_position, final_position)
    # 动作
    action = ["up", "down", "left", "right"]

    # q-learning 算法解决grid world问题
    # q table用于存储动作状态值 三位列表，初始值均为0
    q_table = [[[0 for j in range(len(world))] for i in range(len(world))] for k in range(4)]
    policy = [[0 for j in range(len(world))] for i in range(len(world))]
    episodes = 700
    alpha = 0.7
    gamma = 0.5
    epsilon = 0.5

    for episode in range(episodes):
        current_state = start_position
        save = [current_state]
        while True:
            # 策略选择动作
            if random.randint(1,100)/100 > epsilon:
                action_index = policy[current_state[0]][current_state[1]]
            else:
                action_index = random.randint(0,3)
            next_state = action_result(action[action_index], current_state, 11)
            reward = get_reward(next_state, final_position, stone_list, current_state)

            # 更新q值表
            maxone, _ = get_maxq(q_table, next_state)
            q_table[action_index][current_state[0]][current_state[1]] += \
                alpha*(reward + gamma*maxone - q_table[action_index][current_state[0]][current_state[1]])
            # 更新策略
            _, argmax = get_maxq(q_table, current_state)
            policy[current_state[0]][current_state[1]] = argmax

            # 时间步长改变
            current_state = next_state
            save.append(current_state)
            if reward == 10 or reward == -10:
                # print(save)
                break

    # 进行推理
    state = start_position
    res = [state]
    print("begin:", state, end=";")
    for i in range(20):
        a_index = policy[state[0]][state[1]]
        next_state = action_result(action[a_index], state, 11)
        print(next_state, end=";")
        res.append(next_state)
        if next_state == final_position:
            print("bingo!")
            print("共走了",i+1,"步")
            plot_world(world, stone_list, start_position, final_position,res)
            print("使用q-learning训练并推理产生的结果图见根目录'qlearning-grid-result.png'")
            print("使用q-learning训练并推理产生的策略已保存在'qlearning-policy.txt'")
            print("q-learning策略：")
            print_policy(policy, stone_list, final_position)
            break
        state = next_state












