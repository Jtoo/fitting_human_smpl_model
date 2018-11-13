
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
    self.covars = tf.constant(gmm["covars"], dtype=tf.float32)
    self.means = tf.constant(gmm["means"], dtype=tf.float32)
    self.weights = tf.constant(gmm["weights"], dtype=tf.float32)
    self.mvns = [tfd.MultivariateNormalFullCovariance(
      loc=mu, covariance_matrix=cov) for (mu, cov) in zip(
        gmm["means"], gmm["covars"])]
    print("mvns.type:", type(self.mvns))

  def __call__(self, pose):
    sketon_pose = pose[:, 3:] # without the global rotation
    log_probs = tf.hstack([mvn.log_prob(sketon_pose) for mvn in self.mvns])
    max_index = tf.argmax(log_probs, axis=1)
    return -self.means[max_index]*log_probs[max_index]


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
    alpha = 1
    angle_prior_weight_default = tf.constant(4.04*1e2, dtype=tf.float32)
    self.angle_prior_weight = tf.placeholder_with_default(
        angle_prior_weight_default, name="angle_prior_weight", shape=[])

    loss_angle = self.angle_prior_weight * alpha * (
        tf.exp(self.poses[:, 55]) + tf.exp(-self.poses[:, 58]) +
        tf.exp(-self.poses[:, 12]) + tf.exp(-self.poses[:, 15]))
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

    # for keypoints
    real_keypoints_2d = tf.placeholder(
      shape=[1, NUM_KEY_JOINTS, 2], dtype=tf.float32, name="real_keypoints_2d")
    weights_keypoints_2d = tf.placeholder(
      shape=[1, NUM_KEY_JOINTS], dtype=tf.float32, name="weights_keypoints_2d")

    weights_keypoints_2d_expanded = tf.expand_dims(weights_keypoints_2d, 2)
    weights_keypoints_2d_expanded = tf.tile(weights_keypoints_2d_expanded, [1, 1, 2])

    self.joints_loss = tf.reduce_mean(
        weights_keypoints_2d_expanded * tf.square(real_keypoints_2d-self.proj_joints_2d), axis=1)

    # dense points error
    index_map = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS], dtype=tf.float32, name="visible_point_index_map")
    img_verts = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS, 2], dtype=tf.float32, name="real_verts_2d")

    index_map_expanded = tf.expand_dims(index_map, 2)
    index_map_expanded = tf.tile(index_map_expanded, [1, 1, 2])

    self.dense_point_loss = tf.reduce_sum(index_map_expanded * tf.square(self.proj_verts_2d - img_verts))
    num_visible_points = tf.reduce_sum(index_map_expanded)
    self.dense_point_loss = 0.01 * tf.divide(self.dense_point_loss, num_visible_points)

    self.angle_prior_loss = self.calulate_angle_prior_loss()
    # self.loss_op = self.dense_point_loss + self.angle_prior_loss
    self.loss_op = self.dense_point_loss

    self.learning_rate = tf.Variable(0.01, trainable=False)

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

    return index_map, img_verts, self.proj_verts_2d


