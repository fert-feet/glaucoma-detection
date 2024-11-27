from utils.drawer import Plotter

draw_tool = Plotter('./logs/image')
# draw_tool.show_train_img('../processed-data/pneumonia/test.npy', '../processed-data/pneumonia/test.csv')
# draw_tool.show_origin_img('../origin-data/pneumonia')
draw_tool.show_layer_output_img('./logs/model/pneumonia_mode_cnn_acc_99.7369409996242.pth')