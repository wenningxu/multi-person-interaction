import pybullet as p
import pybullet_data
import numpy as np
from tqdm import tqdm
import psutil
from multiprocessing import Process

# 初始化 PyBullet


class CollisionDepth(object):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def __init__(self, joints1, joints2, radius=0.02):

        self.joints1 = joints1
        self.joints2 = joints2
        self.radius = radius
        self.body_ids = []

    def add_bones(self, ids):
        self.body_ids += ids

    def clear_scene(self):
        for body_id in self.body_ids:
            p.removeBody(body_id)
        self.body_ids = []

    def calculate_xz_bounding_box(self, pose):
        x_coords = pose[:, 0]
        z_coords = pose[:, 2]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_z, max_z = np.min(z_coords), np.max(z_coords)
        return min_x, max_x, min_z, max_z

    def check_bounding_box_overlap(self, pose1, pose2):
        bbox1 = self.calculate_xz_bounding_box(pose1)
        bbox2 = self.calculate_xz_bounding_box(pose2)
        min_x1, max_x1, min_z1, max_z1 = bbox1
        min_x2, max_x2, min_z2, max_z2 = bbox2

        # 检查是否在 x 或 z 方向完全分离
        if max_x1 < min_x2 or max_x2 < min_x1:
            return False  # x 方向分离
        if max_z1 < min_z2 or max_z2 < min_z1:
            return False  # z 方向分离

        return True
    def check_depth(self):
        collision_depth_all = 0
        for frame in tqdm(range(len(self.joints1))):
            if not self.check_bounding_box_overlap(self.joints1[frame], self.joints2[frame]):
                continue
            collision_depth_all += self.check_depth_frame(frame)

        return collision_depth_all

    def check_depth_frame(self, frame):
        try:
            p1_ids = self.create_skeleton(self.joints1[frame])
            p2_ids = self.create_skeleton(self.joints2[frame])
            self.add_bones(p1_ids)
            self.add_bones(p2_ids)

            collisions_depth = 0
            for bone1 in p1_ids:
                for bone2 in p2_ids:
                    contact_points = p.getClosestPoints(bone1, bone2, distance=0.0)
                    if contact_points:
                        for contact in contact_points:
                            penetration_depth = contact[8]
                            if (bone1 in p1_ids and bone2 in p2_ids) or (bone1 in p2_ids and bone2 in p1_ids):
                                collisions_depth += penetration_depth

            self.clear_scene()
        except Exception as e:
            print(f"Error in subprocess for frame {frame}: {e}")
            return 0

        return collisions_depth

    def create_skeleton(self, poses):
        ids = []
        for chain in kinematic_chain:
            for i in range(len(chain) - 1):
                start_joint = poses[chain[i]]
                end_joint = poses[chain[i + 1]]
                id = self.create_bone_mesh(start_joint, end_joint, self.radius)
                ids.append(id)

        return ids

    def create_bone_mesh(self, start, end, radius=0.02):
        """
        创建骨骼的网格表示为细长的盒子
        :param start: 起点坐标 (x, y, z)
        :param end: 终点坐标 (x, y, z)
        :param radius: 骨骼的半径
        :return: PyBullet 碰撞和视觉形状 ID
        """
        # 骨骼长度和方向
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        length = np.linalg.norm(direction)
        center = (start + end) / 2  # 骨骼中点

        # 创建细长盒子表示骨骼
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[radius, radius, length / 2]
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[radius, radius, length / 2],
            rgbaColor=[1, 0, 0, 1]  # 红色骨骼
        )

        # 计算旋转矩阵
        if length > 0:
            direction /= length
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            rotation_angle = np.arccos(np.dot(z_axis, direction))
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis /= np.linalg.norm(rotation_axis)
                rotation = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)
            else:
                rotation = [0, 0, 0, 1]
        else:
            rotation = [0, 0, 0, 1]

        # 创建刚体
        bone_id = p.createMultiBody(
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center,
            baseOrientation=rotation
        )
        return bone_id

    def check_memory_limit(self, limit_gb):
        mem = psutil.virtual_memory()
        if mem.used > limit_gb * 1024 ** 3:
            raise MemoryError(f"Memory usage exceeded {limit_gb} GB")



kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# 遍历关节点对，生成骨骼
if __name__ == '__main__':

    p1 = np.load('../results/multi/punch_5p_wo_gli_p0.npy')
    p2 = np.load('../results/multi/punch_5p_wo_gli_p4.npy')
    collision_checker = CollisionDepth(p1, p2)
    depth = collision_checker.check_depth()
    print(depth)