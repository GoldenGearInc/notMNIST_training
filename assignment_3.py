from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt


pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    print('Train set:', train_dataset.shape, train_labels.shape)
    print('Valid set:', valid_dataset.shape, valid_labels.shape)
    print('Test set:', test_dataset.shape, test_labels.shape)
    

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size * image_size).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def accuracy(prediction, labels):
    return(100 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))
           / labels.shape[0])    
    

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

'''    
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regul = tf.placeholder(tf.float32)
    
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf_train_labels)) + beta_regul * tf.nn.l2_loss(weights)
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(
            tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(
            tf_test_dataset, weights) + biases)
    

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    
    for step in range(num_steps):
        offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
        
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        regul = 1e-3
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,
                     beta_regul : regul}
        
        _, l, prediction = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss: %d at step %f' % (l, step))
            print('Minibatch accuracy: %.1f%%' % accuracy(prediction, batch_labels))
            print('Valid accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
            print('Test accuracy %.1f%%' % accuracy(
                    test_prediction.eval(), test_labels))

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
    
batch_size = 128
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta_regul = tf.placeholder(tf.float32)
    
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
    biases2 = tf.Variable(tf.zeros([num_labels]))
    
    lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    drop1 = tf.nn.dropout(lay1_train, 0.5)
    logits = tf.matmul(drop1, weights2) + biases2
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf_train_labels) + beta_regul * \
        (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)))
    
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    valid_lay1 = tf.nn.relu(tf.matmul(valid_dataset, weights1) + biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_lay1, weights2) + biases2)
    test_lay1 = tf.nn.relu(tf.matmul(test_dataset, weights1) + biases1)
    test_prediction = tf.nn.softmax(tf.matmul(test_lay1, weights2) + biases2)


num_steps = 3001

#regul_params = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]
regul_params = [1e-3]
valid_scores = []
train_scores = []

for regul in regul_params:
    all_train_acc = []
    all_valid_acc = []
    all_steps = []
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        
        for step in range(num_steps):
            offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
            
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,
                         beta_regul : regul}
            
            _, l, prediction = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                train_acc = accuracy(prediction, batch_labels)
                valid_acc = accuracy(valid_prediction.eval(), valid_labels)
                test_acc = accuracy(test_prediction.eval(), test_labels)
                
                all_train_acc.append(train_acc)
                all_valid_acc.append(valid_acc)
                all_steps.append(step)
                print('Minibatch loss: %d at step %f' % (l, step))
                print('Minibatch accuracy: %.1f%%' % train_acc)
                print('Valid accuracy: %.1f%%' % valid_acc)
                print('Test accuracy %.1f%%' % test_acc)

    axis = plt.gca()
    axis.set_ylim([0, 100])
    plt.plot(all_steps, all_train_acc, linewidth=2.0, c='r', label='train')
    plt.plot(all_steps, all_valid_acc, linewidth=2.0, c='b', label='valid')
    plt.legend()
    plt.show()
 
    train_scores.append(all_train_acc[len(all_train_acc) - 1])
    valid_scores.append(all_valid_acc[len(all_valid_acc) - 1])
    
print('Regularizarion graph')
axis = plt.gca()
axis.set_ylim([0, 100])
plt.plot(regul_params, train_scores, linewidth=2.0, c='r', label='train')
plt.plot(regul_params, valid_scores, linewidth=2.0, c='b', label='valid')
plt.legend()
plt.show()
'''
    
batch_size = 128
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 512
num_hidden_nodes3 = 256

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(
            batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(
            batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)
    
    weights1 = tf.Variable(tf.truncated_normal(
            [image_size * image_size, num_hidden_nodes1], stddev=np.sqrt(
                    2.0 / (image_size * image_size))))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
    weights2 = tf.Variable(tf.truncated_normal(
            [num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
    weights3 = tf.Variable(tf.truncated_normal(
            [num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
    weights4 = tf.Variable(tf.truncated_normal(
            [num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)))
    biases4 = tf.Variable(tf.zeros([num_labels]))
    
    train_lay1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    train_lay2 = tf.nn.relu(tf.matmul(train_lay1, weights2) + biases2)
    train_lay3 = tf.nn.relu(tf.matmul(train_lay2, weights3) + biases3)
    logits = tf.matmul(train_lay3, weights4) + biases4
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf_train_labels))
    
    learning_rate = tf.train.exponential_decay(0.5, global_step, 4000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)
    
    train_prediction = tf.nn.softmax(logits)
    valid_lay1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_lay2 = tf.nn.relu(tf.matmul(valid_lay1, weights2) + biases2)
    valid_lay3 = tf.nn.relu(tf.matmul(valid_lay2, weights3) + biases3)
    valid_prediction = tf.matmul(valid_lay3, weights4) + biases4
    test_lay1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_lay2 = tf.nn.relu(tf.matmul(test_lay1, weights2) + biases2)
    test_lay3 = tf.nn.relu(tf.matmul(test_lay2, weights3) + biases3)
    test_prediction = tf.matmul(test_lay3, weights4) + biases4
    
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        
        _, l, prediction = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d : %f' % (step, l))
            print('Train accuracy: %.1f%%' % accuracy(prediction, batch_labels))
            print('Valid accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
            print('Test accuracy %.1f%%' % accuracy(
                    test_prediction.eval(), test_labels))
    
    
    
    
    
    






















