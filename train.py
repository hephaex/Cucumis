from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import inference, loss, train
from input_data import get_input_pipeline
import time
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 100, '')
tf.app.flags.DEFINE_integer('max_steps', 40000, '')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, '')
tf.app.flags.DEFINE_string('restore', None, '')
tf.app.flags.DEFINE_string('job_dir', './job','')
tf.app.flags.DEFINE_float('keep_prob', 0.5, '')
tf.app.flags.DEFINE_integer('input_height', '72', '')
tf.app.flags.DEFINE_integer('input_width', '24', '')
tf.app.flags.DEFINE_integer('input_ch', '1','')
tf.app.flags.DEFINE_string('dataset_dir', './dataset', '')

def main(_):
	with tf.Graph().as_default():
		with tf.Session() as sess:
			images, labels, lengths, widths, areas = get_input_pipeline(FLAGS.batch_size, 'train', FLAGS.input_height, FLAGS.input_width, FLAGS.input_ch, root=FLAGS.dataset_dir)

			logits = inference(images, lengths, widths, areas, FLAGS.keep_prob, True)
			cross_entropy = loss(logits, labels)

			train_op = train(cross_entropy, FLAGS.learning_rate)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			if FLAGS.restore is not None:
				saver.restore(sess, FLAGS.restore)

			summary_op = tf.summary.merge_all()
			summary_writer = tf.summary.FileWriter(FLAGS.job_dir + '/summary', sess.graph)

			start_time = time.time()
			try:
				for step in range(FLAGS.max_steps):
					sess.run(train_op)
					sys.stdout.write('\r>>Step:%d'%(step))
					sys.stdout.flush()

					if (step+1)%100==0:
						summary_writer.add_summary(summary_op.eval(), step)
					if (step+1)%500==0:
						duration = time.time() - start_time
						ce = sess.run(cross_entropy)
						print('\rStep %d: Cross Entropy=%g (%.3f sec)'%(step, ce, duration))
						saver.save(sess, FLAGS.job_dir + '/model.ckpt', global_step=step)
						start_time = time.time()

			finally:
				coord.request_stop()
				coord.join(threads)

if __name__ == '__main__':
	tf.app.run()
