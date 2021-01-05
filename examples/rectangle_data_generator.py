import numpy as np


from pystatplottools.pytorch_data_generation.data_generation.datageneratorbaseclass import DataGeneratorBaseClass


class RectangleGenerator(DataGeneratorBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inp_size = kwargs.pop("dim")
        self.tar_size = 2

        self.h, self.w = self.inp_size
        self.rectangle_hs = np.arange(1, self.h)
        self.rectangle_ws = np.arange(1, self.w)

    def sampler(self):
        rect_h = np.random.choice(self.rectangle_hs)
        rect_w = np.random.choice(self.rectangle_ws)
        sample = np.zeros((self.h, self.w))

        left_idx_h = np.random.randint(0, self.h - rect_h + 1, 1)[0]
        left_idx_w = np.random.randint(0, self.w - rect_w + 1, 1)[0]

        sample[left_idx_h:left_idx_h + rect_h, left_idx_w:left_idx_w + rect_w] = np.random.rand() * (rect_h / (left_idx_w + 1)) / self.h
        return sample, np.array([rect_h, rect_w])


class BatchRectangleGenerator(RectangleGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = kwargs.pop("batch_size")

    def sampler(self):
        rect_hs = np.random.choice(self.rectangle_hs, self.batch_size)
        rect_ws = np.random.choice(self.rectangle_ws, self.batch_size)
        batch = np.zeros((self.batch_size, self.h, self.w))

        left_idx_hs, left_idx_ws = np.zeros(self.batch_size, dtype=np.int), np.zeros(self.batch_size, dtype=np.int)
        for rect_h in self.rectangle_hs:
            left_idx_hs[rect_hs == rect_h] = np.random.randint(0, self.h - rect_h + 1, np.sum(rect_hs == rect_h))
        for rect_w in self.rectangle_ws:
            left_idx_ws[rect_ws == rect_w] = np.random.randint(0, self.w - rect_w + 1, np.sum(rect_ws == rect_w))

        for idx, (sample, rect_h, rect_w, left_idx_h, left_idx_w) in enumerate(zip(batch, rect_hs, rect_ws, left_idx_hs, left_idx_ws)):
            batch[idx, left_idx_h:left_idx_h + rect_h, left_idx_w:left_idx_w + rect_w] = np.random.rand() * (rect_h / (left_idx_w + 1)) / self.h

        return batch, np.stack([rect_hs, rect_ws], axis=1)


def data_generator_factory(data_generator_name="RectangleGenerator"):
    if data_generator_name == "BatchRectangleGenerator":
        return BatchRectangleGenerator
    else:
        return RectangleGenerator
