from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import inference 
from input_data import get_input_pipeline
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', '100','')
tf.app.flags.DEFINE_string('restore', None, '')
tf.app.flags.DEFINE_string('job_dir', './job','')
tf.app.flags.DEFINE_integer('input_height', '72', '')
tf.app.flags.DEFINE_integer('input_width', '24', '')
tf.app.flags.DEFINE_integer('input_ch', '1','')
tf.app.flags.DEFINE_string('dataset_dir', './dataset', '')
tf.app.flags.DEFINE_integer('test_data_size', 8000, 'Num of test data')

LABELS = ['2L', 'L', 'M', 'S', '2S', 'BL', 'BM', 'BS', 'C']

def main(_):
	with tf.Graph().as_default():
		with tf.Session() as sess:
			images, labels, lengths, widths, areas = get_input_pipeline(FLAGS.batch_size, 'validation', FLAGS.input_height, FLAGS.input_width, FLAGS.input_ch, root=FLAGS.dataset_dir)
		
			logits = tf.nn.softmax(inference(images, lengths, widths, areas, 1.0, False))
			predictions = tf.argmax(logits, 1)
			accuracy, _ = tf.metrics.accuracy(labels, predictions, updates_collections=['metrics_update'])
			recalls = []
			precisions = []
			for i in xrange(len(LABELS)):
				recall, _ = tf.metrics.recall_at_k(tf.cast(labels, tf.int64), logits, 1, class_id=i, updates_collections=['metrics_update'])
				precision, _ = tf.metrics.precision(tf.equal(labels, i), tf.equal(predictions, i), updates_collections=['metrics_update'])
				recalls.append(recall)
				precisions.append(precision)
			update_op = tf.get_default_graph().get_collection('metrics_update')
		
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
			#sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			saver = tf.train.Saver()
			saver.restore(sess, FLAGS.restore)
		
			start_time = time.time()
			try:
				n = FLAGS.test_data_size//FLAGS.batch_size
				corrects = 0
				for i in xrange(n):
					sess.run(update_op)
				print('Accuracy: %g'%(accuracy.eval()))
				for i in xrange(len(LABELS)):
					print('Recall-%2s: %g'%(LABELS[i], recalls[i].eval()))
				for i in xrange(len(LABELS)):
					print('Precision-%2s: %g'%(LABELS[i], precisions[i].eval()))
		
			finally:
				coord.request_stop()
			coord.join(threads)

if __name__ == '__main__':
	tf.app.run()
