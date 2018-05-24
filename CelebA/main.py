
import tensorflow as tf
import numpy as np
import glob
import utils
import config
import time
import os

from model import VQVAE
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  #
  save_dir = os.path.join(args.save_dir, args.model_type)
  img_dir = os.path.join(args.img_dir, args.model_type)
  log_dir = os.path.join(args.log_dir, args.model_type)
  train_dir = args.train_dir

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  summary_writer = tf.summary.FileWriter(log_dir)
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  model = VQVAE(args, sess, name="vqvae")

  img_paths = glob.glob('data/img_align_celeba/*.jpg')
  train_paths, test_paths = train_test_split(img_paths, test_size=0.1, random_state=args.random_seed)
  celeba = utils.DiskImageData(sess, train_paths, args.batch_size, shape=[218, 178, 3])
  total_batch = celeba.num_examples // args.batch_size

  for epoch in range(1, args.nb_epoch + 1):
    print "Epoch %d start with learning rate %f" % (epoch, model.learning_rate.eval(sess))
    print "- " * 50
    epoch_start_time = time.time()
    step_start_time = epoch_start_time
    for i in range(1, total_batch + 1):
      global_step = sess.run(model.global_step)
      x_batch = celeba.next_batch()

      _, loss, rec_loss, vq, commit, global_step, summaries = model.train(x_batch)
      summary_writer.add_summary(summaries, global_step)

      if i % args.print_step == 0:
        print "epoch %d, step %d, loss %f, rec_loss %f, vq_loss %f, commit_loss %f, time %.2fs" \
            % (epoch, global_step, loss, rec_loss, vq, commit, time.time()-step_start_time)
        step_start_time = time.time()

    if args.anneal and epoch >= args.anneal_start:
      sess.run(model.lr_decay_op)

    if epoch % args.save_epoch == 0:
      x_batch = celeba.next_batch()
      x_recon = model.reconstruct(x_batch)
      utils.save_images(x_batch, [10, 10], os.path.join(img_dir, "rawImage%s.jpg" % epoch))
      utils.save_images(x_recon, [10, 10], os.path.join(img_dir, "reconstruct%s.jpg" % epoch))

  model.saver.save(sess, os.path.join(save_dir, "model.ckpt"))
  print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)