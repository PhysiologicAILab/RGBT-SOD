import numpy as np
import os
from utils_downstream.test_data import test_dataset
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from utils_downstream.config import dutrgbd,njud,nlpr,stere,sip,rgbd135,ssd,lfsd
from utils_downstream.config import RGBD_SOD_Models
from tqdm import tqdm


test_datasets = {'VT821':dutrgbd,'VT1000':nlpr,'VT5000':nlpr}

for method_name,method_map_root in RGBD_SOD_Models.items():
  
    method_name = 'LSNet_baseline_ssl10'
    
    for name, root in test_datasets.items():       
        
        root = '/home/yclab/guangyu/Segmentation/data/RGBT_dataset/test/'
        root = '{}/{}/GT'.format(root, name)
        
        method_map_root='{}/{}/{}'.format('./SOD',method_name,name)
        print(name)
        sal_root = method_map_root
        print(sal_root)
        gt_root = root
        print(gt_root)
        if os.path.exists(sal_root):
            test_loader = test_dataset(sal_root, gt_root)
            mae,fm,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()
            for i in tqdm(range(test_loader.size)):
                # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = sal.size
                    gt = gt.resize((x, y))
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res)
                if res.max() == res.min():
                    res = res/255
                else:
                    res = (res - res.min()) / (res.max() - res.min())


                mae.update(res, gt)
                sm.update(res,gt)
                fm.update(res, gt)
                em.update(res,gt)
                wfm.update(res,gt)
                m_dice.update(res,gt)
                m_iou.update(res,gt)
                ber.update(res,gt)
                acc.update(res,gt)

            MAE = mae.show()
            maxf,meanf,_,_ = fm.show()
            sm = sm.show()
            em = em.show()
            wfm = wfm.show()
            m_dice = m_dice.show()
            m_iou = m_iou.show()
            ber = ber.show()
            acc = acc.show()
            print('method_name: {} dataset: {} MAE: {:.4f} Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} M_dice: {:.4f} M_iou: {:.4f} Acc: {:.4f}'.format(method_name,name, MAE,ber, maxf,meanf,wfm,sm,em, m_dice, m_iou,acc))
