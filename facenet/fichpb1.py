import tensorflow.compat.v1 as tf

meta_path = 'D:/bureau1/facenett/faceRecognition-yolo-facenet/models/20210427-195211/model-20210427-195211.meta' # Your .meta file
output_node_names = ['output:0']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/models/20210427-195211/model-20210427-195211.meta'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())