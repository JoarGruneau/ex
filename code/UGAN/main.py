import os
from tf_unet import unet
from tf_unet import image_util
from tf_unet import image_gen

if __name__ == '__main__':
    net = unet.Unet(layers=1, features_root=4, cost="IoU", channels=3, n_class=2, summaries=False)
    data_provider = image_util.ImageDataProvider("images/*.png", border_size = 0, data_suffix='.png', mask_suffix='_mask.png')
    #generator = image_gen.RgbDataProvider(1024, 1024, cnt=20, rectangles=False)
    trainer = unet.Trainer(net, batch_size=1, optimizer='momentum',
                           opt_kwargs={'momentum': 0.2, "learning_rate": 0.2, "decay_rate": 0.9})
    path = trainer.train(data_provider, "train/", dropout=0.75,
                         training_iters=1771, epochs=25, display_step=5,  restore=False)
    x_test = a._load_file("images/00000000.png")
    x_test = a._process_data([x_test])
    prediction = net.predict(path, x_test)