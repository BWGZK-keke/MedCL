import math
import sys
import random
import time
import datetime
from typing import Iterable
import torch
import torchvision
import torch.nn.functional as Func
import PIL
import numpy as np
import util.misc as utils
from inference import keep_largest_connected_components
import torch.nn.functional as F
import torchvision.transforms.functional as Func

def one_hot(gt):
    gt_flatten = gt.flatten().astype(np.uint8)
    gt_onehot = np.zeros((gt_flatten.size, 5))
    gt_onehot[np.arange(gt_flatten.size), gt_flatten] = 1
    return gt_onehot.reshape(1, gt.shape[1], gt.shape[2], 5)

def get_params(degrees):
    angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
    return angle

def self_aug(samples, targets):
    samples = samples.tensors
    num_samples = samples.shape[0]
    all_samples = []
    all_labels = []
    for index in range(num_samples):
        img = samples[index]
        target = targets[index]

        aug_imgs = []
        aug_labels = []
        for i in range(2):
            angle1 = get_params([0,360])
            angle2 = get_params([0,360])
            # img = img.copy()
            rotated_img = Func.rotate(img, angle1, PIL.Image.Resampling.NEAREST)
            rotated_img_aux = Func.rotate(img, angle2, PIL.Image.Resampling.NEAREST)
            mask = target['masks']
            rotated_mask = Func.rotate(mask, angle1, PIL.Image.Resampling.NEAREST)
            rotated_mask_aux = Func.rotate(mask, angle2, PIL.Image.Resampling.NEAREST)
            target_size = random.randint(int(0.9*img.shape[1]), img.shape[1])
            x_min = random.randint(0, img.shape[1]-target_size)
            x_max = x_min+target_size
            y_min = random.randint(0, img.shape[1]-target_size)
            y_max = y_min+target_size
            indicator = torch.zeros_like(rotated_img)
            indicator[:,x_min:x_max,y_min:y_max] = 1
            # beta = np.random.beta(0.5, 0.5, [1,1,1])
            beta = torch.tensor(np.random.beta(0.5, 0.5, [1,1,1]))
            mix_img = indicator*rotated_img + (1-beta)*(1-indicator)*rotated_img_aux+beta*(1-indicator)*rotated_img
            rotated_mask_onehot = to_onehot_dim5(rotated_mask)
            rotated_mask_aux_onehot = to_onehot_dim5(rotated_mask_aux)
            indicator = indicator.unsqueeze(0)
            beta = beta.unsqueeze(0)
            mix_label = indicator*rotated_mask_onehot + (1-beta)*(1-indicator)*rotated_mask_aux_onehot+beta*(1-indicator)*rotated_mask_onehot
            aug_imgs.append(mix_img)
            aug_labels.append(mix_label)
        aug_imgs = torch.concatenate(aug_imgs,0)
        aug_labels = torch.concatenate(aug_labels,0)
        all_samples.append(aug_imgs.unsqueeze(0))
        all_labels.append(aug_labels.unsqueeze(0))
    all_samples= torch.cat(all_samples,0)
    all_labels = torch.cat(all_labels,0)
    return all_samples, all_labels

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def augment(x, l, device, beta=0.5):
    mixs = []
    try:
        x=x.tensors
    except:
        pass
    mix = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    mix = torch.maximum(mix, 1 - mix)
    mix = mix.to(device)
    mixs.append(mix)
    xmix = x * mix + torch.flip(x,(0,)) * (1 - mix)
    lmix = l * mix + torch.flip(l,(0,)) * (1 - mix)
    return xmix, lmix, mixs

def mix_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, rates = augment(samples, target_masks, device)
    return aug_samples, aug_targets, rates

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def convert_targets_values(targets):
    indicator = torch.zeros((len(targets),4,212,212)).cuda()
    for index in range(len(targets)):
        t = targets[index]
        values = t["lab_values"]
        values = [i for i in values if i <4]
        for value in values:
            indicator[index,value,:,:] += 1
    return indicator

def Cutout_augment(x, l, device, beta=1):
    lams = []
    try:
        x=x.tensors
    except:
        pass
    lam = torch.distributions.beta.Beta(beta, beta).sample([x.shape[0], 1, 1, 1])
    bboxs = []
    x_flip = torch.flip(x,(0,))
    l_flip = torch.flip(l,(0,))
    for index in range(x.shape[0]):
        bbx1, bby1, bbx2, bby2= rand_bbox(x.shape, lam[index,0,0,0])
        x[index,:,bbx1:bbx2,bby1:bby2] = 0
        l[index,:,bbx1:bbx2,bby1:bby2]= 0
        bboxs.append([bbx1, bby1, bbx2, bby2])
    return x, l, bboxs

def Cutout_targets(samples, targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    aug_samples, aug_targets, bboxs = Cutout_augment(samples, target_masks, device)
    return aug_samples, aug_targets, bboxs

def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim5(target_masks):
    target_masks = target_masks.unsqueeze(0)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks[0]

def rotate(imgs,labels):
    num = imgs.shape[0]
    imgs_out_list = []
    labels_out_list = []
    angles = []
    
    for i in range(num):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
        
        rotated_img = torchvision.transforms.functional.rotate(img, angle, PIL.Image.Resampling.NEAREST, False, None)
        rotated_label = torchvision.transforms.functional.rotate(label, angle, PIL.Image.Resampling.NEAREST, False, None)
        
        imgs_out_list.append(rotated_img)
        labels_out_list.append(rotated_label)
        
        angles.append(angle)
    
    imgs_out = torch.stack(imgs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, labels_out, angles

def flip(imgs, labels):
    imgs_list = []
    labels_list = []
    flips = []
    for i in range(imgs.shape[0]):
        img = imgs[i,:,:,:]
        label = labels[i,:,:,:]

        flipped_img = img
        flipped_label = label

        flip_choice = int(random.random()*4)
        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])
            flipped_label = torch.flip(flipped_label,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])
            flipped_label = torch.flip(flipped_label,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])
            flipped_label = torch.flip(flipped_label,[1,2])

        flips.append(flip_choice)
        imgs_list.append(flipped_img)
        labels_list.append(flipped_label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    return imgs_out, labels_out, flips

def flip_back(outputs, flips):
    outs = []
    for i in range(outputs["pred_masks"].shape[0]):
        output = outputs["pred_masks"][i,:,:,:]
        flip_choice = flips[i]
        flipped_img = output

        if flip_choice == 0:
            pass

        if flip_choice == 1:
            flipped_img = torch.flip(flipped_img,[1])

        if flip_choice == 2:
            flipped_img = torch.flip(flipped_img,[2])

        if flip_choice == 3:
            flipped_img = torch.flip(flipped_img,[1,2])

        outs.append(flipped_img)
        outs = torch.stack(outs)
        return {"pred_masks":outs}

def rotate_back(outputs,angles):
    num = outputs["pred_masks"].shape[0]
    outputs_out_list = []
    
    for i in range(num):
        output = outputs["pred_masks"][i,:,:,:]
        angle = -angles[i]
        
        rotated_output =  torchvision.transforms.functional.rotate(output, angle, PIL.Image.Resampling.NEAREST, False, None)
        
        outputs_out_list.append(rotated_output)
    
    outputs_out = torch.stack(outputs_out_list) 
    return {"pred_masks":outputs_out}

def Cutout(imgs,labels, device, n_holes=1, length=32):
    labels = [t["masks"] for t in labels]
    labels = torch.stack(labels)

    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []
    masks_list = []

    for i in range(num):
        label = labels[i,:,:,:]
        img = imgs[i,:,:,:]
        mask = np.ones((1, h, w), np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask
        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		
        z_i = F.normalize(emb_i, dim=1)  
        z_j = F.normalize(emb_j, dim=1)  

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size*5)
        return loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, estimate_alpha):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items()}
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()

    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ]
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        indicator = convert_targets_values(targets)
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start

        all_samples, all_labels = self_aug(samples, targets)
        samples = all_samples.to(device)
        targets = all_labels.to(device)
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets] 

        samples = torch.flatten(samples, start_dim=0, end_dim=1)
        samples = samples.unsqueeze(1)
        targets = torch.flatten(targets, start_dim=0, end_dim=1)
        indicator = indicator.unsqueeze(0).repeat((2,1,1,1,1))
        indicator = torch.flatten(indicator, start_dim=0, end_dim=1)
        

        # indicator = convert_targets_values(targets)
        # targets = convert_targets(targets, device)
        beta = torch.tensor(np.random.beta(0.5, 0.5, [samples.shape[0],1,1,1])).cuda()
        mixed_samples = beta*samples + (1-beta)*torch.flip(samples,[0])
        mixed_samples = mixed_samples.float()
        samples = samples.float()
        mixed_targets = beta*targets + (1-beta)*torch.flip(targets,[0])
            
        outputs = model(samples, task)
        mix_outputs = model(mixed_samples, task)
        pred_mix = beta*outputs["pred_masks"]+ (1-beta)*torch.flip(outputs["pred_masks"],[0])
        
        y_labeled_mixed = mixed_targets[:,0:4,:,:]
        ce_mixed = -y_labeled_mixed*torch.log(mix_outputs["pred_masks"]+1e-10)
        ce_mixed_loss = ce_mixed.sum()/(y_labeled_mixed.sum()+1e-10)
        
        flat_mix_pred = mix_outputs["pred_masks"].reshape(samples.shape[0],4,-1)
        flat_pred_mix = pred_mix.reshape(samples.shape[0],4,-1)
        mix_consistency = 1 - F.cosine_similarity(flat_pred_mix,flat_mix_pred,dim=2).mean()
        exist_loss = -torch.log((indicator*outputs["pred_masks"]).sum(1)+1e-10).mean()
        
        features = outputs["individual_code"]
        features = features.reshape((32,2,5,128))
        feature_flat1 = features[:,0].flatten(start_dim=1, end_dim=2)
        feature_flat2 = features[:,1].flatten(start_dim=1, end_dim=2)
        loss_func = ContrastiveLoss(batch_size=32)
        loss_contra = loss_func(feature_flat1, feature_flat2)

        features = outputs["anatomy_code"]
        centers = features.mean(0, keepdims=True)
        nom = torch.exp(F.cosine_similarity(features, centers, dim=2))
        anatomy_nom = torch.exp(F.cosine_similarity(centers[:,4:5], centers[:,1:4].sum(1, keepdims=True), dim=2))

        class_num = features.shape[1]
        dem = 0
        for i in range(class_num):
            if i < class_num - 1:
                for j in range(i+1, class_num, 1):
                    dem += torch.exp(F.cosine_similarity(centers[0,i:i + 1,:], centers[0, j:j + 1,:], dim=1))

        contrastive_loss = -torch.log(((nom+anatomy_nom)/(nom+anatomy_nom+dem))+1e-10)
        contrastive_loss = contrastive_loss.mean()
        
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in ['loss_CrossEntropy',"loss_multiDice"])

        # reduce losses over all GPUs for logging purposes
        loss_dict["contrastive_loss"] = contrastive_loss
        loss_dict["exist_loss"] = exist_loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items() if k in ['loss_CrossEntropy',"contrastive_loss","loss_multiDice"]}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        if step == 0:
            # print("Q loss:", subloss)
            print("Positive loss:", losses.item())
            print("contrastive loss:", contrastive_loss.item())
            print("exist loss:", exist_loss.item())
            print("ce mixed loss:", ce_mixed_loss.item())
            print("mix consistency:", mix_consistency.item())
            print("loss constra:", loss_contra.item())


        final_losses = losses + contrastive_loss + exist_loss + ce_mixed_loss + mix_consistency + loss_contra

        optimizer.zero_grad()
        final_losses.backward()
        # for name, p in model.named_parameters():
        #     if "prototype" in name:
        #         p.grad = None
        optimizer.step()

        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
    # gather the stats from all processes  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, dataloader_dict, device, output_dir, visualizer, epoch, writer):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('loss_multiDice', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    print_freq = 10
    numbers = { k : len(v) for k, v in dataloader_dict.items() }
    iterats = { k : iter(v) for k, v in dataloader_dict.items() }
    tasks = dataloader_dict.keys()
    counts = { k : 0 for k in tasks }
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    for step in range(total_steps):
        start = time.time()
        tasks = [ t for t in tasks if counts[t] < numbers[t] ] 
        task = random.sample(tasks, 1)[0]
        samples, targets = next(iterats[task])
        counts.update({task : counts[task] + 1 })
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        targets_onehot= convert_targets(targets,device)
        outputs = model(samples.tensors, task)

        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict.keys()}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step % round(total_steps / 7.) == 0:  
            ##original  
            sample_list.append(samples.tensors[0])
            ##
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            
            ##original
            target_list.append(targets[0]['masks'])
            ##
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    writer.add_scalar('avg_loss', stats['loss_CrossEntropy'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    
    return stats