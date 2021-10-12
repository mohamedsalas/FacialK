from tensorflow.python.tools import freeze_graph

def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename + '.ckpt')
        self.saver.save(self.sess, filepath)
        return filepath

def save_as_pb(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save check point for graph frozen later
        ckpt_filepath = self.save(directory=directory, filename=filename)
        pbtxt_filename = filename + '.pbtxt'
        pbtxt_filepath = os.path.join(directory, pbtxt_filename)
        pb_filepath = os.path.join(directory, filename + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.
        tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=directory, name=pbtxt_filename, as_text=True)

        # Freeze graph
        # Method 1
        frozen_graph_def=freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names='cnn/output', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
        
        # Method 2
        '''
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = ['cnn/output']

        output_graph_def = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_node_names)
        # For some models, we would like to remove training nodes
        # output_graph_def = graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        '''
        
        with open(pb_filepath, 'wb') as f:
          f.write(frozen_graph_def.SerializeToString())

save_as_pb(self, "./models/20210427-195211/model-20210427-195211.ckpt-15", "20210427-195211")