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
    #
    #
    net = unet.Ugan(cost="cross_entropy", channels=3, n_class=2, border_addition=6, patch_size=1000, summaries=True, unet_kwargs=unet_kwargs)
    # # data_provider = image_util.ImageDataProvider("Potsdam/RGB/*.tif", "Potsdam/Labels", patch_size=1000, border_size=20,
    # #                                              data_suffix="_RGB.tif", mask_suffix='_label.tif',
    # #                                              channels=3, n_class=6, load_saved=False)
    # # data_provider = image_util.ImageDataProvider('Potsdam/train_RGB/', patch_size=1000, border_size=20,
    # #                                              data_suffix="_RGB.tif", mask_suffix='_label.tif',
    # #                                              channels=3, n_class=6, load_saved=True)
    # # data_provider.save_patches('Potsdam/train_RGB/')
    # print(net.offset)
    data_provider = \
        image_util.ImageDataProvider("Potsdam/RGB/*.tif", "Potsdam/bin_labels_resized/", patch_size=1000,
                                                 channels=3, n_class=2, border_size=net.offset//2+6, data_suffix=".tif",
                                                 mask_suffix='_label_mask.tif', weight_suffix ='_weight_map.tif')
    eval_data_provider = \
        image_util.ImageDataProvider("Potsdam/RGB/top_potsdam_6_10.tif", "Potsdam/bin_labels_resized/", patch_size=1000,
                                                 channels=3, n_class=2, border_size=net.offset//2+6, data_suffix=".tif",
                                                 mask_suffix='_label_mask.tif', weight_suffix ='_weight_map.tif')
    #
    learning_opts = {'learning_rate':1e-3}
    trainer = unet.Trainer(net, batch_size=1, optimizer='adam', g_opt_kwargs=learning_opts)
    # # # #                      opt_kwargs={'momentum': 0.9, "learning_rate": 0.2, "decay_rate": 0.9})
    path = trainer.train(data_provider, eval_data_provider, "summaries/", dropout=1.0,
                            training_iters=2, eval_iters=1, epochs=800, display_step=50, predict_step=100,  restore=False)
    # x_test = a._load_file("images/00000000.png")
    # x_test = a._process_data([x_test])
    #path = "summaries/model/model.cpkt-156"
    # net.predict(path, data_provider, test_iters=6, border_size=net.offset//2+6, patch_size=1000,
    #             input_size=1000, name='eval_soft', prediction_path='prediction',
    #             verification_batch_size=1,  combine=False, hard_prediction=False, filter_size=0)
    print('evaluating network')
    net.predict(path, data_provider, test_iters=2, border_size=net.offset//2+6, patch_size=1000,
                input_size=1000, name='eval_hard', prediction_path='prediction',
                verification_batch_size=1, combine=True, hard_prediction=True, evaluate_scores=False, overlay=True, filter_size=0)