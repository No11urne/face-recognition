from models.train import (
    train_origin_model,
    train_model_with_adversarial,
    train_autoencoder,
    train_autoencoder_and_model,
    training_pca,
    train_simple_model_after_resnet
)

# train resnet50 on vgg_face2
############################################################################
# loss_name = 'npair'
# loss_name = 'triplet'
# loss_name = 'arcface'
# loss_name = 'sphereface'
#train_origin_model(loss_name=loss_name,
#                   num_epochs=1,
#                   batch_size=4,
#                   query_size=0.75,
#                   model_name='origin')
############################################################################


# train resnet50 on vgg_face2 & adversarial images
############################################################################
#attack_name = 'pgd'
##attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'

#loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#train_model_with_adversarial(attack_name=attack_name,
#                              loss_name=loss_name,
#                              num_epochs=100,
#                              batch_size=64,
#                              query_size=0.75,
#                              model_name='resnet_with_adversarial')
############################################################################

# train autoencoder
############################################################################
#attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
#train_autoencoder(attack_name=attack_name,
#                   num_epochs=50,
#                   batch_size=32,
#                   model_name='autoencoder')
############################################################################

# train autoencoder and resnet50
############################################################################
#attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
#
#loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#train_autoencoder_and_model(attack_name=attack_name,
#                             loss_name=loss_name,
#                             num_epochs=30,
#                             batch_size=32,
#                             query_size=0.75,
#                             model_name='autoencoder_and_model')
############################################################################
#
#
# train pca
############################################################################
#loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'

# n_components = 200
#n_components = 400
# # n_components = 600
# # n_components = 800

#attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'

#training_pca(loss_name=loss_name,
#              n_components=n_components,
#              attack_name=attack_name,
#              batch_size=32,
#              model_name='pca')
############################################################################

# train simple model after resnet50
############################################################################
# loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
# attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'

# train_simple_model_after_resnet(attack_name,
#                                 loss_name,
#                                 num_epochs=30,
#                                 batch_size=32,
#                                 query_size=0.75,
#                                 model_name='simple_model_after_resnet')
############################################################################
