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
cmdargs = """MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/suji/glove MODEL.PRETRAINED_DETECTOR_CKPT pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/suji/checkpoints/causal-motifs-sgcls-exmp"""
cmdargs = cmdargs.split()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
lmdb_path="vcr_detection"
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
data_loaders_val =  make_vcr_data_loader(cfg, is_distributed=False)
dataset_names = cfg.DATASETS.VCR
device = torch.device("cuda")
idslist = []
for dataset_name, data_loader in zip(dataset_names, data_loaders_val):
    model.eval()
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    with env.begin(write=True) as txn:
        for _, batch in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                images, targets, indexes ,img_ids, img_infos= batch
                result, proposals, output, orig_features= model(images.to(device), None, objdet = True)
                #item_df = pd.DataFrame(columns =['prop_bbox', 'pred_logit', 'objectness', 'bbox_info', 'feat', 'img_info'])
                img_id = img_ids[0]
                img_info = img_infos[0]
                proposal = proposals[0]
                objectness = proposal.get_field('objectness').cpu().numpy()
                bbox_info = {'width': proposal.size[0], 'height': proposal.size[1]}
                predict_logits = proposal.get_field('predict_logits').cpu().numpy()
                prop_bboxs = proposal.bbox.cpu().numpy()
                orig_feat = orig_features[0].cpu().numpy()
                '''print('prop', proposals)
                print('of', orig_feat)
                print("assert len", len(prop_bboxs), len(orig_feat))'''
                assert len(prop_bboxs) == len(orig_feat)
                heap = []
                for prop_bbox,pred_log, objness, feat in  zip(prop_bboxs, predict_logits, objectness, orig_feat):
                    if len(heap) > 20:
                        if heap[0][0] < objness:
                            heapq.heapreplace(heap, (objness,pred_log[0], {'prop_bbox': prop_bbox, 'pred_logit': pred_log, 'objectness':objness, 'bbox_info': bbox_info, 'feat': feat, 'img_info': img_info}))
                    else:
                        heapq.heappush(heap, (objness,pred_log[0], {'prop_bbox': prop_bbox, 'pred_logit': pred_log, 'objectness':objness, 'bbox_info': bbox_info, 'feat': feat, 'img_info': img_info}))
                    #pred_scores = res.get_field('pred_scores').cpu().numpy().tolist()
                    #prop_bbox = prop_bbox.bbox.cpu().numpy()
                    #item_df = item_df.append({'prop_bbox': prop_bbox, 'pred_logit': pred_log, 'objectness':objness, 'bbox_info': bbox_info, 'feat': feat, 'img_info': img_info}, ignore_index=True)
                    #txn.put(img_id.encode('utf-8'), pickle.dumps({'pred_scores': pred_scores, 'bbox': bbox,'bbox_info': bbox_info, 'feat': out, 'img_info': im_info}))
                    #txn.put(img_id.encode('utf-8'), pickle.dumps({'bbox': bbox,'bbox_info': bbox_info, 'feat': out, 'img_info': img_info}))
                #item_df = item_df.sort_values(by=['objectness'])
                idslist.append(img_id)
                txn.put(img_id.encode('utf-8'), pickle.dumps([ent[2] for ent in heap]))
                #txn.put(img_id.encode('utf-8'), pickle.dumps(item_df[:20]))
        txn.put("keys".encode('utf-8'), pickle.dumps(idslist))
    torch.cuda.empty_cache()
