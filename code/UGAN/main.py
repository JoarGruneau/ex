import os
from tf_unet import unet
from tf_unet import image_util
from tf_unet import image_gen
from PIL import Image
import numpy as np
#five layers 1196 or 1212 borders (92+6) or (92+14)
#five layers 1132 borders(92+2)



# noinspection PyPackageRequirements
if __name__ == '__main__':
    unet_kwargs = {'layers':5, 'features_root':16}
    resnet_kwargs={'init_num_filters':64, 'block_sizes': [2, 2, 2], 'block_strides': [1, 2, 2],}


    net = unet.Ugan(cost="cross_entropy", channels=3, n_class=2, border_addition=6, patch_size=1000, summaries=True, unet_kwargs=unet_kwargs, resnet_kwargs=resnet_kwargs)
    # data_provider = image_util.ImageDataProvider("Potsdam/RGB/*.tif", "Potsdam/Labels", patch_size=1000, border_size=20,
    #                                              data_suffix="_RGB.tif", mask_suffix='_label.tif',
    #                                              channels=3, n_class=6, load_saved=False)
    # data_provider = image_util.ImageDataProvider('Potsdam/train_RGB/', patch_size=1000, border_size=20,
    #                                              data_suffix="_RGB.tif", mask_suffix='_label.tif',
    #                                              channels=3, n_class=6, load_saved=True)
    # data_provider.save_patches('Potsdam/train_RGB/')
    print(net.offset)
    data_provider = \
        image_util.ImageDataProvider("Potsdam/resized2/train/*.tif", "Potsdam/bin_labels_resized/", patch_size=1000,
                                                 channels=3, n_class=2, border_size=net.offset//2+6, data_suffix="_RGB.tif",
                                                 mask_suffix='_label_mask_mask.tif')
    eval_data_provider = \
        image_util.ImageDataProvider("Potsdam/resized2/eval/*.tif", "Potsdam/bin_labels_resized/", patch_size=1000,
                                                 channels=3, n_class=2, border_size=net.offset//2+6, data_suffix="_RGB.tif",
                                                 mask_suffix='_label_mask_mask.tif', shuffle_data=False)
    # data_provider = image_util.ImageDataProvider("images/*.png", border_size=0, data_suffix=".png",
    #                                              mask_suffix='_mask.png')
    # #data_provider = image_gen.RgbDataProvider(1024, 1024, cnt=20, rectangles=False)
    d_learning_opts={'learning_rate':1e-3,'beta1':0.5}
    g_learning_opts = {'learning_rate': 1e-4, 'beta1': 0.5}
    trainer = unet.Trainer(net, batch_size=1, optimizer='adam', d_opt_kwargs=d_learning_opts, g_opt_kwargs=g_learning_opts)
    # # #                      opt_kwargs={'momentum': 0.9, "learning_rate": 0.2, "decay_rate": 0.9})
    path = trainer.train(data_provider, eval_data_provider, "summaries/", cut_off = 1.0, check_discriminator=20, dropout=0.75,
                            training_iters=20, eval_iters=4, epochs=800, display_step=10, predict_step=50,  restore=True)
    # x_test = a._load_file("images/00000000.png")
    # x_test = a._process_data([x_test])
    # prediction = net.predict(path, x_test)