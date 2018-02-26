import os
from tf_unet import unet
from tf_unet import image_util

if __name__ == '__main__':
    net = unet.Unet(layers=3, features_root=4, cost="dice_coefficient", cost_kwargs={"class_weights": [0.7, 99.3]}, channels=3, n_class=2)
    data_provider = image_util.ImageDataProvider("images/*.png", net.offset//2, data_suffix='.png', mask_suffix='_mask.png')
    trainer = unet.Trainer(net, batch_size=1, optimizer='momentum',
                           opt_kwargs={'momentum': 0.5, "learning_rate": 0.02, "decay_rate": 1})
    path = trainer.train(data_provider, "train/", dropout=0.5,
                         training_iters=100, epochs=25, display_step=50,  restore=False)
    x_test = a._load_file("images/00000000.png")
    x_test = a._process_data([x_test])
    prediction = net.predict(path, x_test)