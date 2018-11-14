
import pickle
import numpy as np
import tensorflow as tf
from tf_smpl.projection import batch_orth_proj_idrot
from tf_smpl.batch_smpl import SMPL

import tensorflow_probability as tfp
tfd = tfp.distributions

NUM_SMPL_VERTS = 6890
NUM_KEY_JOINTS = 19

class MaxMixtureCompletePrior:
  def __init__(self, gaussians_params_path, n_gaussians=8, prefix=3):
    self.n_gaussians = n_gaussians
    self.prefix = prefix
    self.create_prior_from_cmu(gaussians_params_path)

  def create_prior_from_cmu(self, gaussians_params_path):
    gmm = pickle.load(open(gaussians_params_path, 'r'))
    self.covars = gmm["covars"].astype(np.float32)   # shape: (8, 69, 69)
    self.means = gmm["means"].astype(np.float32)     # shape: (8, 69)
    self.weights = gmm["weights"].astype(np.float32) # shape: (8,)

    self.mvns = [tfd.MultivariateNormalFullCovariance(
      loc=mu, covariance_matrix=cov) for (mu, cov) in zip(self.means, self.covars)]
    self.mix_gauss = tfd.Mixture(
        cat=tfd.Categorical(probs=self.weights), components=self.mvns)
    print("dtype(self.mvns):", self.mvns[0].dtype)
    print("dtype(self.mix_gauss):", self.mix_gauss.dtype)

  def __call__(self, pose):
    # pose.shape: [N, 72]
    sketon_pose = pose[:, 3:] # poses without the global rotation
    print("dtype(sketon_pose):", sketon_pose.dtype)
    loss = -self.mix_gauss.log_prob(sketon_pose)
    loss = tf.reduce_sum(loss) # sum over batches, here we only use 1 batch
    return loss


class Model:
  def __init__(self, smpl_model_path, gaussians_params_path):
    self.smplModel = SMPL(smpl_model_path)
    self.gussians_prior = MaxMixtureCompletePrior(gaussians_params_path)

  def calulate_angle_prior_loss(self):
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2

    # angle_prior_weight_default = tf.constant(4.04*1e2, dtype=tf.float32)
    # self.angle_prior_weight = tf.placeholder_with_default(angle_prior_weight_default, name="angle_prior_weight", shape=[])
    loss_angle = (tf.exp(self.poses[:, 55]) + tf.exp(-self.poses[:, 58]) +
                  tf.exp(-self.poses[:, 12]) + tf.exp(-self.poses[:, 15]))
    loss_angle = tf.reduce_sum(loss_angle) # sum over batches, here we only use 1 batch
    return loss_angle


  def build_model(self):
    """
    pose shape: 3 x 23 + 3 = 72
    shapes shape: 10
    batch size fixed as 1
    """
    with tf.variable_scope('PoseParams'):
      self.poses = tf.Variable(
          np.zeros((1, 72)), name="poses", dtype=tf.float32)
      self.shapes = tf.Variable(
          np.zeros((1, 10)), name="shapes", dtype=tf.float32)

    with tf.variable_scope('CameraParams'):
      # camera, shape: [1, 3], value per sample: [s/f, tx, ty]
      initial_cams = np.zeros([1, 3])
      initial_cams[0] = [200, 0.4, 0.5]
      self.cams = tf.Variable(
          initial_cams, name='cameras', dtype=tf.float32)

    self.verts_3d, self.joints_3d, _ = self.smplModel(self.shapes, self.poses, get_skin=True)
    self.proj_verts_2d = batch_orth_proj_idrot(self.verts_3d, self.cams, name='proj_verts_2d')
    self.proj_joints_2d = batch_orth_proj_idrot(self.joints_3d, self.cams, name='proj_verts_2d')

    print("proj_verts_2d.shape", self.proj_verts_2d.shape)

    # keypoints error
    self.real_keypoints_2d = tf.placeholder(
      shape=[1, NUM_KEY_JOINTS, 2], dtype=tf.float32, name="real_keypoints_2d")
    self.weights_keypoints_2d = tf.placeholder(
      shape=[1, NUM_KEY_JOINTS], dtype=tf.float32, name="weights_keypoints_2d")

    # weights_keypoints_2d_expanded = tf.expand_dims(self.weights_keypoints_2d, 2)
    # weights_keypoints_2d_expanded = tf.tile(weights_keypoints_2d_expanded, [1, 1, 2])

    self.joints_loss = tf.reduce_mean(
        self.weights_keypoints_2d * tf.norm(self.real_keypoints_2d-self.proj_joints_2d, ord=2, axis=2))

    # dense points error
    self.index_map = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS], dtype=tf.float32, name="visible_point_index_map")
    self.img_verts = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS, 2], dtype=tf.float32, name="real_verts_2d")

    # index_map_expanded = tf.expand_dims(self.index_map, 2)
    # index_map_expanded = tf.tile(index_map_expanded, [1, 1, 2])

    self.dense_point_loss = tf.reduce_sum(self.index_map * tf.norm(self.proj_verts_2d - self.img_verts, ord=2, axis=2))
    self.num_visible_points = tf.reduce_sum(self.index_map)
    self.dense_point_loss = tf.divide(self.dense_point_loss, self.num_visible_points)

    # angle of limbs loss
    self.angle_prior_loss = self.calulate_angle_prior_loss()
    # poses prior loss
    self.gussians_prior_loss = self.gussians_prior(self.poses)
    # shapes regulization (mean shapes are all zero)
    self.shapes_loss = tf.reduce_sum(self.shapes)

    self.all_losses = tf.stack(
        [self.joints_loss, self.angle_prior_loss, self.gussians_prior_loss, self.shapes_loss, self.dense_point_loss])
    print("dtype(self.all_losses):", self.all_losses.dtype, self.all_losses.shape)
    # losses weights
    self.losses_weights = tf.placeholder(
      shape=[5], dtype=tf.float32, name="losses_weights")

    self.all_losses_weighted = self.losses_weights * self.all_losses

    # self.loss_op = self.joints_loss + self.angle_prior_loss + self.gussians_prior_loss
    self.loss_op = self.all_losses_weighted

    self.learning_rate = tf.Variable(0.001, trainable=False)

    # minimize the loss wrt cameras, poses and shapes
    optimizer_all = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op_all = optimizer_all.minimize(
        self.loss_op, var_list=[self.cams, self.poses, self.shapes])

    optimizer_cams = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op_cams = optimizer_cams.minimize(
        self.loss_op, var_list=[self.cams])

    optimizer_poses = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op_poses = optimizer_poses.minimize(
        self.loss_op, var_list=[self.poses])

    optimizer_both_cams_poses = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train_op_both_cams_poses = optimizer_both_cams_poses.minimize(
        self.loss_op, var_list=[self.cams, self.poses])

    return self.index_map, self.img_verts, self.proj_verts_2d


