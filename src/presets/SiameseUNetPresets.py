import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
import tools.ModuleTrainer as MT
import tools.DataEvaluation as DE
import tools.reporters.QQEmailReporter as QQ

from datasets.Common import RunningMode
from models.SiameseUNet import SiameseUNet
from datasets.LEVIRCDDataset import LEVIRCDDataset


def train_SiameseUNet(out_path: str = None):
    if out_path is None:
        raise ValueError(out_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=SiameseUNet(),
                               save_frequency=10, pt_path=out_path, epoch=600, batch_size=6)

    trainer.report_loss = lambda loss, epoch: QQ.send_myself_QQEmail(
        "SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))

    trainer.train()

    torch.save(trainer.module, out_path+"\\SiameseUNet_Finished.pt")


def train_semi_finished_SiameseUNet(pt_path: str = None, out_path: str = None):
    if pt_path is None:
        raise ValueError(pt_path)

    if out_path is None:
        raise ValueError(out_path)

    module = torch.load(pt_path, map_location=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=module,
                               save_frequency=10, pt_path=out_path, epoch=600, batch_size=6)

    trainer.report_loss = lambda loss, epoch: QQ.send_myself_QQEmail(
        "SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))

    trainer.train()

    torch.save(trainer.module, out_path+"\\SiameseUNet_Finished.pt")


def evaluate_SiameseUNet(pt_path: str = None, save_predict_img: bool = False, img_path: str = None):
    if pt_path is None:
        raise ValueError(pt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(pt_path, map_location=device)
    model = model.eval()

    split_num = 2

    dataloader = data.DataLoader(dataset=LEVIRCDDataset(
        running_mode=RunningMode.evaluation, split_num=split_num), shuffle=False)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    step = 0
    sum_count = dataloader.__len__()
    img_block_cache = []

    for input, label in dataloader:
        input = input.to(device)

        output_mtx = torch.squeeze(model(input)).cpu().detach().numpy()
        label_mtx = torch.squeeze(label).detach().numpy()

        pixel_count = output_mtx.shape[0]

        for i in range(pixel_count):
            for j in range(pixel_count):
                input_positive = output_mtx[i][j] > 0.5
                label_positive = label_mtx[i][j] > 0.5

                # if input and label are the same class
                if not (input_positive ^ label_positive):

                    # if positive
                    if input_positive:
                        tp += 1
                    else:
                        tn += 1

                else:
                    if input_positive:
                        fp += 1
                    else:
                        fn += 1

        step += 1

        if save_predict_img:
            if img_path is None:
                raise ValueError(img_path)

            img_block_cache.append(output_mtx)

            if step % (split_num**2) is 0:
                x1 = np.hstack([img_block_cache[0], img_block_cache[1]])
                x2 = np.hstack([img_block_cache[2], img_block_cache[3]])
                x = np.vstack([x1, x2])

                img = Image.fromarray((x*255).astype(np.uint8))
                img.save(img_path+"\{}.bmp".format(step))

                img_block_cache = []

                ValueError()

        print("{} of {} imgs has been calculated.".format(step, sum_count))

    DSC, mIOU, recall, precision, f1, OA = DE.evaluate_indicators(tp, fp, tn, fn)

    print("tp = {}, tn = {}, fp = {}, fn = {}".format(tp, tn, fp, fn))
    print("DSC = {}, mIOU = {}, recall = {}, precision = {}, f1 = {}, OA = {}".format(
        DSC, mIOU, recall, precision, f1, OA))
