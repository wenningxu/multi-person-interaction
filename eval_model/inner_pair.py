import os
import sys
sys.path.append(sys.path[0]+r"/../")
import torch
from itertools import combinations
from configs import get_config
from tqdm import tqdm
import math
import trimesh
from trimesh.collision import CollisionManager
import smplx

from datasets import get_dataset_motion_loader, get_motion_loader
from models import *
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from eval_model.collision import *
from utils.preprocess import *

def extract_pair(samples):
    # samples (person_num, frame_num, 262)

    pairs = list(combinations(samples, 2))
    pairs = [torch.cat(pair, dim=1) for pair in pairs]
    pairs = torch.stack(pairs, dim=0)
    return pairs


def extract_split(samples):
    # samples (person_num, frame_num, 262)

    pairs = list(combinations(samples, 2))
    pairs1 = [pair[0] for pair in pairs]
    pairs2 = [pair[1] for pair in pairs]
    pairs1 = torch.cat(pairs1, dim=0)
    pairs2 = torch.cat(pairs2, dim=0)
    return pairs1, pairs2

def extract_pairs_all(samples_dict):
    # samples (sample_num, person_num, frame, 262)
    pair_dict = OrderedDict({})
    for model_name, samples in samples_dict.items():
        paris_all = []
        for sample in samples:
            paris_all.append(extract_pair(sample))

        paris_all = torch.cat(paris_all, dim=0)

        pair_dict[model_name] = paris_all

    return pair_dict



def evaluate_fid(sudo_gt_dict, motion_dict, file):
    inner_dict = OrderedDict({})
    print('========== Evaluating InnerFID ==========')
    # print(gt_mu)
    for model_name, motions in motion_dict.items():
        with torch.no_grad():
            sudo_gt_motion = sudo_gt_dict[model_name.split('_')[-1]]
            gt_motion_embeddings = eval_wrapper.get_motion_embeddings_from_sample(sudo_gt_motion).cpu().numpy()
            gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
            motion_embeddings = eval_wrapper.get_motion_embeddings_from_sample(motions).cpu().numpy()
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        inner_dict[model_name] = fid
        print(f'---> [{model_name}] InnerFID: {fid:.4f}')
        print(f'---> [{model_name}] InnerFID: {fid:.4f}', file=file, flush=True)


def evaluate_collision(motion_dict, file):
    collision = OrderedDict({})
    print('========== Evaluating MultiCollisionDepth Distance ==========')
    for motion_loader_name, motions in motion_dict.items():
        B, T = motions.shape[:2]
        motions = motions.reshape(B, T, 2, -1)
        joints1 = motions[..., 0, :66].reshape(-1, 22, 3)
        joints2 = motions[..., 1, :66].reshape(-1, 22, 3)
        collision_detector = CollisionDepth(joints1.numpy(), joints2.numpy())
        depth = abs(collision_detector.check_depth()) / B
        collision[motion_loader_name] = depth
        print(f'---> [{motion_loader_name}] MultiCollisionDepth: {depth:.4f}')
        print(f'---> [{motion_loader_name}] MultiCollisionDepth: {depth:.4f}', file=file, flush=True)

    return collision


def evaluate_mesh(vertice_folder, log, p_n=3, ablation=0):
    BODY_MODEL = 'joints2smpl/smpl_models'
    model = smplx.create(BODY_MODEL, model_type='smpl')
    collision = OrderedDict({})
    print('========== Evaluating MultiCollisionDepth Distance ==========')
    files = os.listdir(vertice_folder)
    vertices_all1 = []
    vertices_all2 = []
    all_size = 0
    all_contact = 0
    contact_frame = 0

    for file in tqdm(files):
        if ('p1' not in file):
            continue
        if ablation==0:
            if ('wo' in file) or ('free' in file):
                continue
        elif ablation==1:
            if 'all' not in file:
                continue
        elif ablation==2:
            if 'unrelated' not in file:
                continue
        elif ablation==3:
            if 'gli' not in file:
                continue
        elif ablation==-1:
            if 'free' not in file:
                continue
        else:
            print('no this set')

        index = 2 if ablation<=0 else 4
        if ablation == -2:
            person_num = 2
        else:
            person_num = int(file.split('_')[index][0])
        if person_num!=p_n:
            continue
        vertives = []
        for p in range(person_num):
            new_name = file.replace('p1', f'p{p+1}')
            vertice_path = os.path.join(vertice_folder, new_name)
            vertice = np.load(vertice_path)
            vertives.append(torch.from_numpy(vertice))
        vertives_pairs1,  vertives_pairs2 = extract_split(vertives)
        vertices_all1.append(vertives_pairs1)
        vertices_all2.append(vertives_pairs2)
        all_size += person_num

    vertices_all1 = torch.cat(vertices_all1, dim=0)
    vertices_all2 = torch.cat(vertices_all2, dim=0)

    depth = 0
    all_frame = len(vertices_all1)
    for frame in tqdm(range(len(vertices_all1))):

        mesh1 = trimesh.Trimesh(-vertices_all1[frame], model.faces)
        mesh2 = trimesh.Trimesh(-vertices_all2[frame], model.faces)

        # 初始化 CollisionManager
        manager = CollisionManager()

        # 添加网格到管理器
        manager.add_object("mesh1", mesh1)
        manager.add_object("mesh2", mesh2)

        # 检查碰撞
        collision, names, contacts = manager.in_collision_internal(
            return_names=True,
            return_data=True
        )

        if collision:
            contact_frame += 1
            for contact in contacts:
                all_contact += 1
                depth += contact.depth

    depth_p_sample = depth / all_size
    depth_p_contacts = depth / all_contact
    depth_p_frame = depth / all_frame
    contact_frame_rate = contact_frame / all_frame
    contact_p_frame = all_contact / all_frame
    contact_p_cframe = all_contact / contact_frame
    print(f'depth_p_sample_{p_n}p: {depth_p_sample:.6f}')
    print(f'depth_p_sample_{p_n}p: {depth_p_sample:.6f}', file=log, flush=True)
    print(f'depth_p_contacts_{p_n}p: {depth_p_contacts:.6f}')
    print(f'depth_p_contacts_{p_n}p: {depth_p_contacts:.6f}', file=log, flush=True)
    print(f'depth_p_frame_{p_n}p: {depth_p_frame:.6f}')
    print(f'depth_p_frame_{p_n}p: {depth_p_frame:.6f}', file=log, flush=True)
    print(f'contact_frame_rate_{p_n}p: {contact_frame_rate:.6f}')
    print(f'contact_frame_rate_{p_n}p: {contact_frame_rate:.6f}', file=log, flush=True)
    print(f'contact_p_frame_{p_n}p: {contact_p_frame:.6f}', file=log, flush=True)
    print(f'contact_p_cframe_{p_n}p: {contact_p_cframe:.6f}', file=log, flush=True)

    return collision





if __name__ == '__main__':
    batch_size = 96
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    # data_cfg = get_config("configs/datasets.yaml").interhuman_test
    # gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    evalmodel_cfg = get_config("configs/eval_model.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    # sudogt_folder = 'results/eval_multi/sudogt'
    # sudogt_list = os.listdir(sudogt_folder)

    # gt = {'3p':[], '4p':[], '5p':[]}
    # for key in gt.keys():
    #     per_num = int(key[0])
    #     com = math.comb(per_num, 2)
    #     samples_gt = []
    #     for file in sudogt_list:
    #         if file.endswith('npy'):
    #             sample_gt = np.load(os.path.join(sudogt_folder, file)).reshape(210, -1)
    #             sample_gt = np.tile(sample_gt, [com, 1, 1])
    #             samples_gt.append(sample_gt)
    #
    #     samples = np.concatenate(samples_gt, axis=0)
    #     gt[key] = torch.from_numpy(samples)


    test = {}
    # test['mingle_3p'] = []
    test['mingle_4p'] = []
    test['mingle_5p'] = []
    sample_fodler = 'results/eval_multi/sample'
    sample_files = os.listdir(sample_fodler)
    for sample_file in sample_files:
        if sample_file.endswith('npy') and ('all' not in sample_file):
            samples_eval = np.load(os.path.join(sample_fodler, sample_file))
            person_n = int(sample_file.split('_')[-1][0])
            test[f'mingle_{person_n}p'].append(torch.from_numpy(samples_eval))

    # test['mingle_3p'] = torch.stack(test['mingle_3p'], dim=0)
    test['mingle_4p'] = torch.stack(test['mingle_4p'], dim=0)
    test['mingle_5p'] = torch.stack(test['mingle_5p'], dim=0)

    test_pair = extract_pairs_all(test)

    file = './mingle_intergen.log'
    with open(file, 'w') as f:
        evaluate_mesh('joints2smpl/demo/eval/inter_gen_eval/vertice', f, 2, ablation=-2)

    # file = './mingle_multi_wo_unrelated.log'
    # with open(file, 'w') as f:
    #     evaluate_mesh('results/eval_multi/smpl/vertices', f, 4, ablation=2)
    #     evaluate_mesh('results/eval_multi/smpl/vertices', f, 5, ablation=2)

