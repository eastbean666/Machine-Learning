"""《RRT算法》
    时间：2025.06.16
    作者：不去幼儿园
"""
import numpy as np
import matplotlib.pyplot as plt
import random
 
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
 
def distance(n1, n2):
    return np.hypot(n1.x - n2.x, n1.y - n2.y)
 
def get_random_node(goal_sample_rate, goal):
    if random.random() < goal_sample_rate:
        return Node(goal.x, goal.y)
    return Node(random.uniform(0, 100), random.uniform(0, 100))
 
def steer(from_node, to_node, extend_length=5.0):
    dist = distance(from_node, to_node)
    theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_x = from_node.x + extend_length * np.cos(theta)
    new_y = from_node.y + extend_length * np.sin(theta)
    new_node = Node(new_x, new_y)
    new_node.parent = from_node
    return new_node

# 定义圆形障碍物列表：每个元素为 (ox, oy, radius)
# 可按需修改或扩展为多个障碍
obstacles = [
    (50.0, 50.0, 15.0),  # 中央障碍
]
 
def is_collision(node):
    # 升级：检查圆形障碍物（全局变量 `obstacles`）
    # 若节点位于障碍内，或节点与其父节点之间的线段与圆相交，认为碰撞
    def line_intersects_circle(x1, y1, x2, y2, cx, cy, r):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.hypot(cx - x1, cy - y1) <= r
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return np.hypot(closest_x - cx, closest_y - cy) <= r

    for (ox, oy, r) in obstacles:
        # 检查节点点是否在圆内
        if np.hypot(node.x - ox, node.y - oy) <= r:
            return True
        # 若有父节点，检查父->当前线段是否与圆相交
        if getattr(node, 'parent', None) is not None:
            if line_intersects_circle(node.parent.x, node.parent.y, node.x, node.y, ox, oy, r):
                return True
    return False
 
def rrt(start, goal, max_iter=500, goal_sample_rate=0.05):
    nodes = [start]
    for _ in range(max_iter):
        rnd = get_random_node(goal_sample_rate, goal)
        nearest = min(nodes, key=lambda n: distance(n, rnd))
        new_node = steer(nearest, rnd)
 
        if not is_collision(new_node):
            nodes.append(new_node)
            if distance(new_node, goal) < 5.0:
                goal.parent = new_node
                nodes.append(goal)
                break
    return nodes
 
def draw_path(last_node):
    path = []
    node = last_node
    while node:
        path.append((node.x, node.y))
        node = node.parent
    path = path[::-1]
    plt.plot([x for x, y in path], [y for x, y in path], '-r')
 
def draw_tree(nodes):
    for node in nodes:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], '-g')

def draw_obstacles(obstacles):
    ax = plt.gca()
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5)
        ax.add_patch(circle)
 
start = Node(10, 10)
goal = Node(90, 90)
 
nodes = rrt(start, goal)
draw_tree(nodes)
draw_path(goal)
draw_obstacles(obstacles)
plt.plot(start.x, start.y, "bs", label="Start")
plt.plot(goal.x, goal.y, "gs", label="Goal")
plt.legend()
plt.grid(True)
plt.axis([0, 100, 0, 100])
plt.title("RRT Path Planning (No Obstacles)")
plt.show()