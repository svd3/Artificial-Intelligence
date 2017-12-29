import tensorflow as tf
from GeneticAlgo import *
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#%%

n_nodes_hl1 = 50
n_nodes_hl2 = 50
batch_size = 5

x = tf.placeholder(tf.float32, [None, 100])
y = tf.placeholder(tf.float32, [None,1])


def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([100, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, 1])),
                    'biases':tf.Variable(tf.random_normal([1]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    output = tf.nn.sigmoid(output)
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(prediction - y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(1000):
                epoch_x, epoch_y = generatebatch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                #print(c)

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)



def generatebatch(batch_size):
    data_x = []
    data_y = []
    for i in range(batch_size):
        genes = randomInit(2)
        [p1,p2] = matchScore(genes[0], genes[1])
        y = float(p1)/20
        data_y.append(y)
        data_x.append(np.reshape(genes, [1,100]))
    data_x = np.reshape(data_x, [batch_size,100])
    data_y = np.reshape(data_y, [batch_size,1])
    return [data_x, data_y]


train_neural_network(x)
