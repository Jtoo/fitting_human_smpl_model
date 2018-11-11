
import numpy as np
import tensorflow as tf
from tf_smpl.projection import batch_orth_proj_idrot
from tf_smpl.batch_smpl import SMPL

NUM_SMPL_VERTS = 6890


class Model:
  def __init__(self, smpl_model_path):
    self.smplModel = SMPL(smpl_model_path)

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

    self.verts_3d, Js, _ = self.smplModel(self.shapes, self.poses, get_skin=True)

    self.verts_3d = self.verts_3d * tf.constant([1, -1, 1], dtype=tf.float32)

    self.proj_verts_2d = batch_orth_proj_idrot(self.verts_3d, self.cams, name='proj_verts_2d')

    index_map = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS], dtype=tf.float32, name="visible_point_index_map")
    img_verts = tf.placeholder(
      shape=[1, NUM_SMPL_VERTS, 2], dtype=tf.float32, name="real_verts_2d")

    index_map_expanded = tf.expand_dims(index_map, 2)
    index_map_expanded = tf.tile(index_map_expanded, [1, 1, 2])

    self.visible_pred_verts = self.proj_verts_2d * index_map_expanded
    self.dense_point_loss = tf.reduce_mean(tf.losses.absolute_difference(self.visible_pred_verts, img_verts))

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


