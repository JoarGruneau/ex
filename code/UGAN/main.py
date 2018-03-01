import os
from tf_unet import unet
from tf_unet import image_util
from tf_unet import image_gen
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # c = image_util.ImageDataProvider
    # img=c._load_file(c,"/home/joar/Documents/ex/code/UGAN/Potsdam/RGB/top_potsdam_2_10_label_mask.tif")
    # # img = Image.open("/home/joar/Documents/ex/code/UGAN/Potsdam/RGB/top_potsdam_2_10_label_mask.tif")
    # # img = np.array(img, dtype=np.float32)
    # print (type(img))
    net = unet.Unet(layers=3, features_root=4, cost="cross_entropy", channels=3, n_class=2, summaries=False)
    data_provider = image_util.ImageDataProvider("Potsdam/resized/*.tif", border_size=0, data_suffix=".tif", mask_suffix='_mask.tif')
    # data_provider = image_util.ImageDataProvider("images/*.png", border_size=0, data_suffix=".png",
    #                                              mask_suffix='_mask.png')
    #data_provider = image_gen.RgbDataProvider(1024, 1024, cnt=20, rectangles=False)
    trainer = unet.Trainer(net, batch_size=1, optimizer='adam')
                   #        opt_kwargs={'momentum': 0.9, "learning_rate": 0.2, "decay_rate": 0.9})
    path = trainer.train(data_provider, "train/", dropout=0.75,
                         training_iters=24, epochs=25, display_step=1,  restore=True)
    # x_test = a._load_file("images/00000000.png")
    # x_test = a._process_data([x_test])
    # prediction = net.predict(path, x_test)