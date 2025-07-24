import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil

class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)

    def plot_multi(self, data, path, caption):
        plot_3d_motion_multi(path, paramUtil.t2m_kinematic_chain, data, title=caption, fps=30, radius=4)


    def generate_one_sample(self, prompt, name):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"] = prompt

        window_size = 210
        motion_output = self.generate_loop(batch, window_size)
        result_path = f"results/{name}.mp4"
        if not os.path.exists("results"):
            os.makedirs("results")

        self.plot_t2m([motion_output[0], motion_output[1]],
                      result_path,
                      batch["prompt"])

    def generate_multi_sample(self, prompt, name, inter_graph, window_size = 300):
        self.model.eval()
        batch = OrderedDict({})
        num_p = len(inter_graph['in'])
        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"] = prompt

        motion_output = self.generate_loop_single(batch, window_size, inter_graph)
        joints3d = motion_output[..., :22 * 3].reshape(motion_output.shape[0], -1, 22, 3)
        joints3d = filters.gaussian_filter1d(joints3d, 1, axis=1, mode='nearest')

        if not os.path.exists("results"):
            os.makedirs("results")
        result_path = f"results/{name}.mp4"
        save_path = f'results/{name}_{num_p}p.npy'
        np.save(save_path, joints3d)

        self.plot_multi(joints3d,
                      result_path,
                      batch["prompt"])

    def generate_loop(self, batch, window_size):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        sequences = [[], []]

        batch["text"] = [prompt]
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())


        for j in range(2):
            motion_output = motion_output_both[:,j]

            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)


        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

    def generate_loop_single(self, batch, window_size, intergraph):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size
        batch['inter_graph'] = intergraph
        batch["text"] = [prompt]
        batch = self.model.forward_test_single(batch)
        motion_output = batch["output"]
        outs = []
        for output in motion_output:
            out = self.normalizer.backward(output.cpu().detach().numpy())
            outs.append(out)

        sequences = np.concatenate(outs, axis=0)
        return sequences


def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model



if __name__ == '__main__':
    # torch.manual_seed(37)
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu", weights_only=False)
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))

    inter_graph_in = infer_cfg.INTER_GRAPH.IN
    inter_graph_out = infer_cfg.INTER_GRAPH.OUT
    inter_graph = {'in': inter_graph_in, 'out': inter_graph_out}

    with open("./prompts.txt") as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]

    for text in texts:
        name = text[:48]
        litmodel.generate_multi_sample(text, name, inter_graph=inter_graph, window_size=300)

