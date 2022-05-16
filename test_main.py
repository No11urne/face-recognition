from models.test import (
    eval_resnet50,
    eval_resnet50_with_adversarial,
    eval_autoencoder,
    eval_autoencoder_resnet50,
    eval_pca,
    eval_simple_model_after_resnet50
)

# eval origin resnet50
############################################################################
#loss_name = 'npair'
loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'

attack_name = 'pgd'
#attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
eval_resnet50(loss_name, attack_name, batch_size=128)
############################################################################

# # eval resnet50 with adversarial images
############################################################################
#loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
#attack_name = 'pgd'
#attack_name = 'deepfool'
## # attack_name = 'fgm'
# # attack_name = 'cw'
#eval_resnet50_with_adversarial(loss_name, attack_name, batch_size=32, model_name='resnet_with_adversarial')
# ############################################################################
#
# # eval autoencoder
# ############################################################################
# loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
# attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
# eval_autoencoder(loss_name, attack_name, batch_size=32, model_name='autoencoder')
# ############################################################################
#
# # eval autoencoder and resnet50
# ############################################################################
# loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
# attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
# eval_autoencoder_resnet50(loss_name, attack_name, batch_size=32, model_name='autoencoder_and_model')
# ############################################################################
#
# # eval pca
# ############################################################################
# loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
# attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
#
# n_components = 200
# # # n_components = 400
# # # n_components = 600
# # # n_components = 800
# eval_pca(loss_name, attack_name, n_components, batch_size=32, model_name='pca')
# ############################################################################
#
# # eval resnet50 and simple model
# ############################################################################
# loss_name = 'npair'
# # loss_name = 'triplet'
# # loss_name = 'arcface'
# # loss_name = 'sphereface'
#
# attack_name = 'pgd'
# # attack_name = 'deepfool'
# # attack_name = 'fgm'
# # attack_name = 'cw'
# eval_simple_model_after_resnet50(loss_name, attack_name, batch_size=32, model_name='simple_model_after_resnet')
# ############################################################################
