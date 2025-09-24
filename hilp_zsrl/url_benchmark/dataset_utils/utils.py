from torch.utils.data import DataLoader

class InfiniteDataLoaderWrapper:
    def __init__(self, dataset, dataloder_cfg, ):
        """
        包装 DataLoader，使其成为一个无限迭代器
        :param dataloader: 原始的 DataLoader 对象
        """
        self.dataloader = DataLoader(dataset, **dataloder_cfg)
        self.dataloader_cfg = dataloder_cfg
        self.dataset = dataset
        # 初始化迭代器
        self.iterator = iter(self.dataloader)
    
    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.dataloader)

    def __iter__(self):
        """
        实现无限迭代器：在每个 epoch 完成时重新创建迭代器
        """
        while True:
            try:
                # 获取下一个batch
                yield next(self.iterator)
            except StopIteration:
                # 如果迭代完毕，重新开始
                self.iterator = iter(self.dataloader)  
    
    def close_loader(self):
        del self.dataloader
        del self.iterator
        self.dataloader = None
        self.iterator = None
        # self.dataloader = DataLoader(self.dataset, **self.dataloader_cfg, worker_init_fn=self.worker_init_fn)
        # self.iterator = iter(self.dataloader)
    
    def init_loader(self):
        if hasattr(self.dataset, "reload_context"):
            self.dataset.reload_context()
        self.dataloader = DataLoader(self.dataset, **self.dataloader_cfg)
        self.iterator = iter(self.dataloader)