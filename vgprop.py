import ast
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
import torch
from tqdm import tqdm
import pandas as pd
from maskrcnn_benchmark.data import *
import sys
import lmdb
import os
import pickle
import heapq
cmdargs = """MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/suji/glove MODEL.PRETRAINED_DETECTOR_CKPT pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/suji/checkpoints/causal-motifs-sgcls-exmp"""
cmdargs = cmdargs.split()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
lmdb_path = "detection_vg"

MAP_SIZE = 1e12
env = lmdb.open(lmdb_path, map_size=MAP_SIZE)

new_path = r'/home/suji/spring20/Scene-Graph-Benchmark.pytorch'
sys.path.append(new_path)


cfg.merge_from_file("/home/suji/spring20/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml")
cfg.merge_from_list(cmdargs)

cfg.freeze()

model = build_detection_model(cfg)
model.to(cfg.MODEL.DEVICE)
# Initialize mixed-precision if necessary
use_mixed_precision = cfg.DTYPE == 'float16'

output_dir = "/home/suji/checkpoints/causal-motifs-sgcls-exmp"
checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT)
data_loaders_val =  make_data_loader(cfg, mode='val', is_distributed=False)
print("len(data_loaders_val)", len(data_loaders_val))
device = torch.device("cuda")
idslist = []
id_set = set()

# change collate batch
model.eval()
cpu_device = torch.device("cpu")
torch.cuda.empty_cache()

with env.begin(write=True) as txn:
    for dataset_name, data_loader in zip(cfg.DATASETS.VAL, data_loaders_val):
        for _, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                images, targets, img_ids, img_info, fn = batch
               
                result, proposals, output, orig_features= model(images.to(device), None, objdet = True)
                print("len(proposals)",len(proposals))
                img_id = img_ids[0]
                #img_info = img_infos[0]
                proposal = proposals[0]
                objectness = proposal.get_field('objectness').cpu().numpy()
                bbox_info = {'width': proposal.size[0], 'height': proposal.size[1]}
                predict_logits = proposal.get_field('predict_logits').cpu().numpy()
                prop_bboxs = proposal.bbox.cpu().numpy()
                orig_feat = orig_features[0].cpu().numpy()
                assert len(prop_bboxs) == len(orig_feat)
                heap = []
                for prop_bbox,pred_log, objness, feat in  zip(prop_bboxs, predict_logits, objectness, orig_feat):
                    if len(heap) > 20:
                        if heap[0][0] < objness:
                            heapq.heapreplace(heap, (objness,pred_log[0], {'fn': fn, 'img_info': img_info, 'prop_bbox': prop_bbox, 'pred_logit': pred_log, 'objectness':objness, 'bbox_info': bbox_info, 'feat': feat}))
                    else:
                        heapq.heappush(heap, (objness,pred_log[0], {'fn': fn, 'img_info': img_info, 'prop_bbox': prop_bbox, 'pred_logit': pred_log, 'objectness':objness, 'bbox_info': bbox_info, 'feat': feat}))
                idslist.append(img_info[0]['image_id'])
                if img_id in id_set:
                    assert False
                id_set.add(img_id)
                txn.put(str(img_info[0]['image_id']).encode('utf-8'), pickle.dumps([ent[2] for ent in heap]))
                #torch.cuda.empty_cache()
        txn.put("keys".encode('utf-8'), pickle.dumps(idslist))
    torch.cuda.empty_cache()

