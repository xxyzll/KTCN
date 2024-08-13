from detectron2.engine.train_loop import HookBase
from detectron2.evaluation import inference_on_dataset, print_csv_format
import detectron2.utils.comm as comm
import torch
import os

class ValidationMap(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.defrost()                  
        self.map_interval = 1000            
        self.cross_interval = 50           
        self.enable_loss_val = False        
        self.start_iter = cfg.EVAL.start_iter 
        
        self.eval_data_loader = None
        self.loss_data_loader = None
        self.evaluator = None
        self.max_metric = {                 
            'AP': 0, 
            'AP50': 0, 
            'Unknown Recall50': 0, 
            'Prev class AP50': 0,
            'Current class AP50': 0,
            'm_cls_recall': 0,
            'm_cls_precision': 0,
        }
 
    def after_step(self):
        if self.enable_loss_val and self.trainer.iter % self.cross_interval == 0 and self.trainer.iter >= self.cross_interval:
            self.get_val_loss()
            
        if self.trainer.iter % self.map_interval == 0 and self.trainer.iter >= self.map_interval and self.trainer.iter >= self.start_iter:
            self.trainer.model.eval()
            self.eval_dataset()
            self.trainer.model.train()

    def get_val_loss(self):
        if self.loss_data_loader is None:
            cfg = self.cfg.clone()
            cfg.DATASETS.TRAIN = self.cfg.DATASETS.TEST
            self.loss_data_loader = iter(self.trainer.build_train_loader(cfg))
            

        with torch.no_grad():
            data = next(self.loss_data_loader)
            loss_dict = self.trainer.model(data)
            loss_dict = self.add_prefix_to_keys(loss_dict, 'val_')
            self.trainer.storage.put_scalars(**dict(loss_dict))
            self.trainer.storage.put_scalar("val_total_loss", sum(loss for loss in loss_dict.values()))
            

    def add_prefix_to_keys(self, dictionary, prefix):
        new_dict = {}
        for key, value in dictionary.items():
            new_key = prefix + str(key)
            new_dict[new_key] = value
        return new_dict
    
    def eval_dataset(self):
        if self.eval_data_loader is None:
            self.eval_data_loader = self.trainer.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
            self.evaluator = self.trainer.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
        result = inference_on_dataset(self.trainer.model,
                                      self.eval_data_loader,
                                      self.evaluator)
        print_csv_format(result)
        self.save_model(result)
        self.log_result(result)
        
    def save_model(self, result):
        """
            result: [{
                'bbox': {
                    'AP': float, 
                    'AP50': float, 
                    'Unknown Recall50': float, 
                    'Prev class AP50': float,
                    'Current class AP50': float} 
            }]
        """
        if not os.path.exists(self.cfg.OUTPUT_DIR):
            os.makedirs(self.cfg.OUTPUT_DIR)
        if 'bbox' not in result.keys():
            return 
            
        for metric_key, metric_val in result['bbox'].items():
            if metric_key in self.max_metric and metric_val> self.max_metric[metric_key]:
                if os.path.exists(os.path.join(self.cfg.OUTPUT_DIR, f'bast_{metric_key}_{self.max_metric[metric_key]}.pth')):
                    os.remove(os.path.join(self.cfg.OUTPUT_DIR, f'bast_{metric_key}_{self.max_metric[metric_key]}.pth'))
                    
                self.max_metric[metric_key] = metric_val
                self.trainer.checkpointer.save(f'bast_{metric_key}_{metric_val}', iteration=self.trainer.iter+1)
                
    def log_result(self, result):
        """
            result: {
                'bbox': {
                    'AP': float, 
                    'AP50': float, 
                    'Unknown Recall50': float, 
                    'Prev class AP50': float,
                    'Current class AP50': float} 
            }
        """
        if 'bbox' not in result.keys():
            return 
        res = result['bbox']
        res = self.add_prefix_to_keys(res, 'val')
        self.trainer.storage.put_scalars(**dict(res))

