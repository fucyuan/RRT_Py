import numpy as np
import matplotlib.pyplot as plt
import random

class RRT:
    def __init__(self, start, goal, map_size, step_size, max_iter, obstacles):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map_size = map_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.obstacles = obstacles

    def get_random_point(self):
        return (random.uniform(0, self.map_size[0]), 
                random.uniform(0, self.map_size[1]))

    def get_nearest_node(self, point):
        distances = [np.linalg.norm(np.array(node.point) - np.array(point))
                    for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_node, to_point):
        direction = np.array(to_point) - np.array(from_node.point)
        distance = np.linalg.norm(direction)
        if distance < 1e-6:  # 避免零向量
            return None
        direction = direction / distance  # 单位化
        step = min(distance, self.step_size)
        new_point = tuple(np.array(from_node.point) + direction * step)
        return Node(new_point, from_node)
    
    def is_collision(self, from_node, to_node):
        for obstacle in self.obstacles:
            if self.is_segment_intersecting_circle(from_node.point, 
                                                 to_node.point, 
                                                 obstacle):
                return True
        return False

    def is_segment_intersecting_circle(self, p1, p2, circle):
        center, radius = circle
        p1 = np.array(p1)
        p2 = np.array(p2)
        center = np.array(center)
        
        # 线段向量计算
        d = p2 - p1
        f = p1 - center
        
        # 二次方程系数
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return False  # 无实根
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # 检查根是否在线段参数范围 [0,1] 内
        return any(0 <= t <= 1 for t in [t1, t2])

    def is_goal_reached(self, node):
        return np.linalg.norm(np.array(node.point) - self.goal.point) <= self.step_size

    def generate(self):
        for _ in range(self.max_iter):
            # 随机采样时10%概率直接采样目标点
            if random.random() < 0.1:
                rand_point = self.goal.point
            else:
                rand_point = self.get_random_point()
            
            nearest_node = self.get_nearest_node(rand_point)
            new_node = self.steer(nearest_node, rand_point)
            
            if new_node and not self.is_collision(nearest_node, new_node):
                self.nodes.append(new_node)
                
                # 尝试连接目标点
                if self.is_goal_reached(new_node):
                    final_node = self.steer(new_node, self.goal.point)
                    if final_node and not self.is_collision(new_node, final_node):
                        self.goal.parent = final_node
                        self.nodes.append(self.goal)
                        return self.get_path()
        return None

    def get_path(self):
        path = []
        current_node = self.goal
        while current_node:
            path.append(current_node.point)
            current_node = current_node.parent
        return path[::-1]  # 反转路径顺序
    
    def draw(self, path=None):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        
        # 绘制所有节点和连接
        for node in self.nodes:
            if node.parent:
                plt.plot([node.point[0], node.parent.point[0]],
                        [node.point[1], node.parent.point[1]], 
                        'gray', linewidth=0.5, alpha=0.5)
        
        # 绘制最终路径
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='Optimal Path')
        
        # 绘制障碍物
        for center, radius in self.obstacles:
            circle = plt.Circle(center, radius, color='black', alpha=0.6)
            ax.add_patch(circle)
        
        # 标记起点终点
        plt.scatter(*self.start.point, c='blue', s=100, label='Start', edgecolors='white')
        plt.scatter(*self.goal.point, c='green', s=100, label='Goal', edgecolors='white')
        
        plt.legend()
        plt.grid(True)
        plt.title('RRT Path Planning')
        plt.show()

class Node:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent

if __name__ == "__main__":
    # 参数配置
    start = (10, 10)
    goal = (90, 90)
    map_size = (100, 100)    # 地图尺寸
    step_size = 5            # 扩展步长
    max_iter = 1000          # 最大迭代次数
    obstacles = [            # 障碍物列表 (中心坐标, 半径)
        ((50, 50), 15),
        ((20, 80), 8),
        ((80, 30), 12),
        ((30, 30), 10),
        ((70, 70), 10)
    ]
    
    # 创建RRT并生成路径
    rrt = RRT(start, goal, map_size, step_size, max_iter, obstacles)
    path = rrt.generate()
    
    if path:
        print("找到路径！路径点序列：")
        for point in path:
            print(f"({point[0]:.1f}, {point[1]:.1f})")
        rrt.draw(path)
    else:
        print("未找到路径，请尝试增加最大迭代次数或调整步长参数")