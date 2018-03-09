import os
from tf_unet import unet
from tf_unet import image_util
from tf_unet import image_gen
from PIL import Image
import numpy as np
#five layers 1196 or 1212 borders (92+6) or (92+14)

# noinspection PyPackageRequirements
if __name__ == '__main__':
    net = unet.Unet(layers=3, features_root=4, cost="cross_entropy", channels=3, n_class=6, border_addition=0, summaries=False)
    # data_provider = image_util.ImageDataProvider("Potsdam/RGB/*.tif", "Potsdam/Labels", patch_size=1000, border_size=20,
    #                                              data_suffix="_RGB.tif", mask_suffix='_label.tif',
    #                                              channels=3, n_class=6, load_saved=False)
    data_provider = image_util.ImageDataProvider('Potsdam/train_RGB/', patch_size=1000, border_size=20,
                                                 data_suffix="_RGB.tif", mask_suffix='_label.tif',
                                                 channels=3, n_class=6, load_saved=True)
    # data_provider.save_patches('Potsdam/train_RGB/')
    # data_provider = image_util.ImageDataProvider("Potsdam/resized/*.tif", patch_size=1024, border_size=net.offset//2 +20, data_suffix=".tif", mask_suffix='_mask.tif')
    # data_provider = image_util.ImageDataProvider("images/*.png", border_size=0, data_suffix=".png",
    #                                              mask_suffix='_mask.png')
    # #data_provider = image_gen.RgbDataProvider(1024, 1024, cnt=20, rectangles=False)
    trainer = unet.Trainer(net, batch_size=1, optimizer='adam')
    # # #                      opt_kwargs={'momentum': 0.9, "learning_rate": 0.2, "decay_rate": 0.9})
    path = trainer.train(data_provider, "train/", dropout=0.75,
                            training_iters=40, epochs=20, display_step=1,  restore=True)
    # x_test = a._load_file("images/00000000.png")
    # x_test = a._process_data([x_test])
    # prediction = net.predict(path, x_test)