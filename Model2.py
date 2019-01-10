import tensorflow as tf 
import create_input
import numpy
import matplotlib.pyplot as plt

input_x=create_input.input_matrix
input_y=create_input.true_output_list
test_x=create_input.test_input_matrix
test_y=create_input.test_true_output_list

n_nodes_hl1 = 500 #First hidden layer has 50 nodes.
n_nodes_hl2 = 500
n_nodes_hl3 = 500
predicted_y=[]

x = tf.placeholder('float',[1,10]) 
y = tf.placeholder('float')

def neural_network_model(data): 
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([10, n_nodes_hl1])),  #A dictionary dataype of python.
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, 1])),
                    'biases':tf.Variable(tf.random_normal([1])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = (tf.matmul(l3,output_layer['weights']) + output_layer['biases'])

    return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_sum((prediction[0][0] - y)**2)
	optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
	hm_epochs = 22
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer=tf.summary.FileWriter('./graphs',sess.graph)
		for epoch in range(hm_epochs):
			epoch_loss = 0
			j=0
			for j in range(len(input_x)):
				epoch_x=input_x[j]
				epoch_y=input_y[j]				
				j, c = sess.run([optimizer, cost], feed_dict={x: [[epoch_x[0],epoch_x[1],epoch_x[2],epoch_x[3],epoch_x[4],epoch_x[5],epoch_x[6],epoch_x[7],epoch_x[8],epoch_x[9]]], y: [epoch_y]})	
				epoch_loss += c
			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
			
			print(prediction.eval({x: [[2000,2100,2200,2300,2400,2500,2600,2700,2800,2900]]}))

		for j in range(len(test_x)):
			epoch_x=test_x[j]
			pred=prediction.eval({x: [[epoch_x[0],epoch_x[1],epoch_x[2],epoch_x[3],epoch_x[4],epoch_x[5],epoch_x[6],epoch_x[7],epoch_x[8],epoch_x[9]]]})
			predicted_y.append(pred[0][0])
        
		for j in range(len(test_y)):
			print(test_y[j],"	",predicted_y[j],"	",test_y[j]-predicted_y[j])
		plt.plot(predicted_y)
		#plt.show()
		plt.plot(test_y)
		plt.show()
		deviation_sum=0
		for j in range(len(test_x)):
			deviation_sum=deviation_sum + ((predicted_y[j]-test_y[j])/test_y[j])**2
		average_deviation=(deviation_sum/len(test_y))**(0.5)
		print("Average Deviation(%): "+str(average_deviation))
        
	writer.close()		
        
train_neural_network(x)
