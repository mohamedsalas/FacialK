import tensorflow.compat.v1 as tf
from keras.models import load_model

saver = tf.train.Saver()
model = keras.models.load_model("emotion_model.hdf5")
sess = keras.backend.get_session()
save_path = saver.save(sess, "model_emotion.ckpt")