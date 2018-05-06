

import tensorflow as tf
import numpy as np
import my_txtutils

# these must match what was saved !
ALPHASIZE = 98# parser ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512


ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("checkpoints/rnn_train_1525462152-6000000.meta")
    #new_saver.restore(sess, tf.train.latest_checkpoint(())
    new_saver.restore(sess,"checkpoints/rnn_train_1525462152-6000000")
  

    x = my_txtutils.convert_from_alphabet(ord("L"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    file = open("generated_output.txt", "w")
    file.write("Hi there, this a generated output poem from the shakespeare machine. have fun! \n\n")
    for i in range(10000):
        yo, h = sess.run(["Yo:0", "H:0"], feed_dict={"X:0": y, "pkeep:0": 1., "Hin:0": h, "batchsize:0": 1})


        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")
        file.write(c)

        if c == "\n":
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            file.write("")
            ncnt = 0
    file.close() 

