import torch
import numpy as np
import ModuleTrainer as MT
import PIL.Image as Image
import torch.utils.data as data

from models.unet import UNet
from datasets.LEVIRCDDataset import LEVIRCDDataset, RunningMode


def train_unet(pt_path: str = None):
    if pt_path is None:
        ValueError(pt_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=UNet(
        in_ch=6, out_ch=1), save_frequency=20, pt_path=pt_path, epoch=400, batch_size=2)
    trainer.train()

    torch.save(trainer.module, pt_path+"\\UNet_Finished.pt")


def evaluate_unet(pt_path: str = None, save_predict_img: bool = False, img_path: str = None):
    if pt_path is None:
        ValueError(pt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(pt_path, map_location=device)
    model = model.eval()

    dataloader = data.DataLoader(dataset=LEVIRCDDataset(
        running_mode=RunningMode.evaluation), shuffle=False)

    # √+
    tp = 0

    # √-
    tn = 0

    # -2+
    fp = 0

    # +2-
    fn = 0

    step = 0
    sum_count = dataloader.__len__()

    for input, label in dataloader:
        input = input.to(device)

        input_mtx = torch.squeeze(model(input)).cpu().detach().numpy()
        label_mtx = torch.squeeze(label).detach().numpy()

        pixel_count = input_mtx.shape[0]

        for i in range(pixel_count):
            for j in range(pixel_count):
                input_positive = input_mtx[i][j] > 0.5
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
                ValueError(img_path)

            img = Image.fromarray(input_mtx.astype(np.uint8))
            img.save(img_path+"\{}.bmp".format(step))

        print("{} of {} imgs has been calculated.".format(step, sum_count))

    DSC = 2*tp/(fp+2*tp+fn)
    mIOU = ((tp/(tp+fp+fn))+(tn/(tn+fn+fp)))/2
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    OA = (tp+tn)/(tp+tn+fp+fn)

    print("tp = {}, tn = {}, fp = {}, fn = {}".format(tp, tn, fp, fn))
    print("DSC = {}, mIOU = {}, recall = {}, precision = {}, f1 = {}, OA = {}".format(
        DSC, mIOU, recall, precision, f1, OA))
