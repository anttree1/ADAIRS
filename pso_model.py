import numpy as np
import random
import test_model as tm
import argparse
import scipy.io as sio
import torch

parser = argparse.ArgumentParser("PSO_autoAugment")
parser.add_argument('--atlas_size', type=int, default=200, help='90,160,200')
parser.add_argument('--W', type=float, default=0.5, help='惯性权重')
parser.add_argument('--C1', type=float, default=1.0, help='个体学习因子')
parser.add_argument('--C2', type=float, default=2.0, help='社会学习因子')
parser.add_argument('--set_size', type=int, default=5, help='选取的集合数量')
parser.add_argument('--population_size', type=int, default=20, help='种群规模')
parser.add_argument('--num_iterations', type=int, default=100, help='迭代次数')

args = parser.parse_args()

load_data = sio.loadmat('/home/ai/data/wangxingy/data/work5/ALLASD1_{}.mat'.format(str(args.atlas_size)))
tr_x = torch.tensor(load_data['net_train']).view(-1, 1, args.atlas_size, args.atlas_size).type(torch.FloatTensor)
ts_x = torch.tensor(load_data['net_test']).view(-1, 1, args.atlas_size, args.atlas_size).type(torch.FloatTensor)
tr_y = load_data['phenotype_train'][:, 2]
ts_y = load_data['phenotype_test'][:, 2]


def generate_element(set_size):
    return [
        random.randint(1, 5),
        random.randint(1, set_size),
        round(random.uniform(0.0, 1.0), 2),
        round(random.uniform(0.0, 1.0), 2)
    ]


def generate_velocity():
    return [
        random.randint(-1, 1),
        random.randint(-1, 1),
        round(random.uniform(-0.1, 0.1), 2),
        round(random.uniform(-0.1, 0.1), 2)
    ]


def csh(population_size):
    particles = []
    for _ in range(population_size):
        position = []
        for _ in range(4):
            position.extend(generate_element(args.set_size))
        velocity = []
        for _ in range(4):
            velocity.extend(generate_velocity())
        particles.append({
            'position': position,
            'velocity': velocity,
            'best_pos': position,
            'fitness': float('inf'),
            'best_fitness': float('inf')
        })
    return particles


def syd(particle):
    position = particle['position']
    fitness = tm.test_main(position, args.atlas_size, tr_x, tr_y, ts_x, ts_y)
    return fitness


def pso(population_size, num_iterations):
    particles = csh(population_size)
    for i in particles:
        print(i['position'])
    print(f"初始化种群：{particles}")
    gBest = None
    gBest_value = float('inf')

    for var_num in range(num_iterations):
        print(f"第{var_num}代种群{particles}")
        for var_p, particle in enumerate(particles):
            print(f"第{var_num}代种群,第{var_p}个个体")
            fitness = syd(particle)
            particle['fitness'] = fitness

            if fitness < particle['best_fitness']:
                particle['best_pos'] = particle['position']
                particle['best_fitness'] = fitness

            if fitness < gBest_value:
                gBest = particle['position']
                gBest_value = fitness

        for particle in particles:
            velocity = particle['velocity']
            position = particle['position']
            best_pos = particle['best_pos']
            for i in range(len(velocity)):
                rd1 = random.random()
                rd2 = random.random()
                velocity[i] = (args.W * velocity[i] +
                               args.C1 * rd1 * (best_pos[i] - position[i]) +
                               args.C2 * rd2 * (gBest[i] - position[i]))
                if i < 2:
                    velocity[i] = int(velocity[i])
            for i in range(len(position)):
                position[i] += velocity[i]
                if i in [0, 4, 8, 12]:
                    position[i] = np.clip(position[i], 0, 5)
                elif i in [1, 5, 9, 13]:
                    position[i] = np.clip(position[i], 1, args.set_size)
                else:
                    position[i] = np.clip(position[i], 0, 1)
                    position[i] = round(position[i], 2)
                if i in [0, 1, 4, 5, 8, 9, 12, 13]:
                    position[i] = int(position[i])

    return gBest, gBest_value


best_position, best_value = pso(args.population_size, args.num_iterations)
print(f"最佳位置：{best_position}")
print(f"最佳适应度值：{best_value}")
