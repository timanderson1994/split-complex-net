import numpy as np
import tensorflow as tf
import layersHDA
from sklearn.utils import shuffle

####
# 
# Code based on code available at: https://github.com/sujaybabruwad/LeNet-in-Tensorflow
#
#####

def runmodel(train, val, test, dataset, BATCH_SIZE, EPOCHS, num_classes = 10, networkSize = 'Baseline', networkType = 'Real', alpha = 1e-3, lam = 0):
    '''
    
    This function is a wrapper used to execute a model for a given algebra type and network size. 
    Inputs:
        train - tuple containing (examples, labels, number of examples) for training set
        validation - tuple containing (examples, labels, number of examples) for validation set
        train - tuple containing (examples, labels, number of examples) for test set
        dataset - name of the dataset being tested. Used when saving the model
        BATCH_SIZE - batch sized used during mini-batch SGD
        num_classes - number of classes in the classifiction task. Default value is 10.
        networkSize - type of network to use. Values are 'Baseline', 'Wide', and 'Deep'
        networkType - algebra type for the layers. Values are 'Real', 'Complex', 'SplitComplex'
        alpha - learning rate for Adam optimizer
        lam - regularization parameter. Default is zero i.e. no regularization
    
    '''
    
    X_train, y_train, NUM_TRAIN = train
    X_validation, y_validation, NUM_VAL = val
    X_test, y_test, NUM_TEST = test
    
    # Set up graph for training
    x = tf.placeholder(tf.float32, (None, 32, 32, X_train.shape[3]))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, num_classes)

    # Select computational graph based on network input
    if networkSize=='Baseline':
        logits = LeNet(x, num_classes = num_classes, inputDim = X_train.shape[3], networkType = networkType)
    elif networkSize=='Wide':
        logits = LeWideNet(x, num_classes = num_classes, inputDim = X_train.shape[3], networkType = networkType)
    elif networkSize=='Deep':
        logits = LeDeepNet(x, num_classes = num_classes, inputDim = X_train.shape[3], networkType = networkType)
    else:
        raise Exception('Invalid network size entered!')
        
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
    
    # Add regularization loss
    params = tf.trainable_variables() 
    reg_loss = tf.add_n([ tf.nn.l2_loss(v) for v in params ])
    
    # Calculate loss
    loss_operation = tf.reduce_mean(cross_entropy) + lam*reg_loss
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    
    training_accuracy = np.zeros(int(np.ceil(EPOCHS*len(X_train)/BATCH_SIZE)))
    validation_accuracy = np.zeros(EPOCHS)
    test_accuracy = np.zeros(EPOCHS)
    
    training_loss = np.zeros(int(np.ceil(EPOCHS*len(X_train)/BATCH_SIZE)))
    validation_loss = np.zeros(EPOCHS)
    test_loss = np.zeros(EPOCHS)
    
    train_loss_count = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
                training_loss[train_loss_count] = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
                training_accuracy[train_loss_count] = sess.run(accuracy_operation, 
                                                               feed_dict={x: batch_x, y: batch_y})
                train_loss_count +=1
            
            
            validation_accuracy[i] = sess.run(accuracy_operation, feed_dict={x: X_validation, y: y_validation}) 
            test_accuracy[i] = sess.run(accuracy_operation, feed_dict={x: X_test, y: y_test}) 
            
            validation_loss[i] = sess.run(loss_operation, feed_dict={x: X_validation, y: y_validation})
            test_loss[i] = sess.run(loss_operation, feed_dict={x: X_test, y: y_test})
            
            print("EPOCH {} ...".format(i+1))
            print("Training Accuracy = {:.3f}".format(training_accuracy[train_loss_count-1]))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy[i]))
            print()
        
        saver.save(sess, 'ExperimentModels/lenet'+'_'+dataset+'_'+networkSize+'_'+networkType)
        print("Model saved")
    
    np.savez('ExperimentData/lenet'+'_'+dataset+'_'+networkSize+'_'+networkType+'_loss.npz', 
             training_loss, validation_loss, test_loss)
    np.savez('ExperimentData/lenet'+'_'+dataset+'_'+networkSize+'_'+networkType+'_acc.npz', 
             training_accuracy, validation_accuracy, test_accuracy)
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./ExperimentModels')) 
        test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_test, y: y_test})
        print("Test Accuracy = {:.3f}".format(test_accuracy))


'''    
def evaluateAcc(X_data, y_data):
    
    num_examples = len(X_data)
    sess = tf.get_default_session()
    total_accuracy = 
    return total_accuracy / num_examples
'''
    
def LeNet(x, inputDim = 1, num_classes = 10, mu = 0, sigma = 0.1, networkType = 'Real'):
    
    if networkType is not 'Real':
        x = layersHDA.typeClone(x, networkType)
    
    conv1 = layersHDA.conv2d(x, weightShape = (5,5,inputDim,6), biasDim = 6, convStride = [1,1,1,1], convPadding = 'VALID')

    # SOLUTION: Activation.
    conv1 = layersHDA.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = layersHDA.avgpool(conv1, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    #conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = layersHDA.conv2d(conv1, weightShape = (5,5,6,16), biasDim = 16, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # ReLU Activation.
    conv2 = layersHDA.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = layersHDA.avgpool(conv2, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    #conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layers 3: Convolutional. Output = 1x1x120
    conv3 = layersHDA.conv2d(conv2, weightShape = (5,5,16,120), biasDim = 120, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # Flatten: Input = 1x1x120. Output = 120.
    fc0 = layersHDA.flatten(conv3)
    
    # Activation.
    fc1 = layersHDA.relu(fc0)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = layersHDA.affine(fc1, weightShape = (120, 84), biasDim = 84)
    
    # Activation.
    fc2 = layersHDA.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3 = layersHDA.affine(fc2, weightShape = (84, num_classes), biasDim = num_classes)
    logits = layersHDA.magnitude(fc3)
    
    return logits

def LeWideNet(x, inputDim = 1, num_classes = 10, mu = 0, sigma = 0.1, networkType = 'Real'):
    
    if networkType is not 'Real':
        x = layersHDA.typeClone(x, networkType)
    
    # Layer 1: Convolutional. Input = 32x32xinputDim. Output = 28x28x9.
    conv1 = layersHDA.conv2d(x, weightShape = (5,5,inputDim,9), biasDim = 9, convStride = [1,1,1,1], convPadding = 'VALID')

    # Activation.
    conv1 = layersHDA.relu(conv1)

    # Pooling. Input = 28x28x9. Output = 14x14x9.
    conv1 = layersHDA.avgpool(conv1, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    
    # Layer 2: Convolutional. Output = 10x10x23.
    conv2 = layersHDA.conv2d(conv1, weightShape = (5,5,9,23), biasDim = 23, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # ReLU Activation.
    conv2 = layersHDA.relu(conv2)

    # Pooling. Input = 10x10x23. Output = 5x5x23.
    conv2 = layersHDA.avgpool(conv2, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    
    # Layers 3: Convolutional. Output = 1x1x170
    conv3 = layersHDA.conv2d(conv2, weightShape = (5,5,23,170), biasDim = 170, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # Flatten: Input = 1x1x170. Output = 170.
    fc0 = layersHDA.flatten(conv3)
    
    # Activation.
    fc1 = layersHDA.relu(fc0)

    # Layer 4: Fully Connected. Input = 170. Output = 119.
    fc2 = layersHDA.affine(fc1, weightShape = (170, 119), biasDim = 119)
    
    # Activation.
    fc2 = layersHDA.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3 = layersHDA.affine(fc2, weightShape = (119, num_classes), biasDim = num_classes)
    logits = layersHDA.magnitude(fc3)
    
    return logits


def LeDeepNet(x, inputDim = 1, num_classes = 10, mu = 0, sigma = 0.1, networkType = 'Real'):
    
    if networkType is not 'Real':
        x = layersHDA.typeClone(x, networkType)
    
    #############
    # Layer 1: Convolutional. Input = 32x32xinputDim. Output = 32x32x6.
    conv1 = layersHDA.conv2d(x, weightShape = (5,5,inputDim,6), biasDim = 6, convStride = [1,1,1,1], convPadding = 'SAME')

    # Activation.
    conv1 = layersHDA.relu(conv1)
    
    # Layer 1b: Convolutional. Input = 32x32xinputDim. Output = 32x32x6.
    conv1b = layersHDA.conv2d(conv1, weightShape = (5,5,6,6), biasDim = 6, convStride = [1,1,1,1], convPadding = 'VALID')

    # Activation.
    conv1b = layersHDA.relu(conv1b)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1b = layersHDA.avgpool(conv1b, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    
    #############
    # Layer 2: Convolutional. Input = 14x14x6. Output = 14x14x16.
    conv2 = layersHDA.conv2d(conv1b, weightShape = (5,5,6,16), biasDim = 16, convStride = [1,1,1,1], convPadding = 'SAME')
    
    # ReLU Activation.
    conv2 = layersHDA.relu(conv2)
    
    # Layer 2b: Convolutional. Output = 10x10x16.
    conv2b = layersHDA.conv2d(conv2, weightShape = (5,5,16,16), biasDim = 16, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # ReLU Activation.
    conv2b = layersHDA.relu(conv2b)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2b = layersHDA.avgpool(conv2b, sizes = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    
    
    ###############
    # Layers 3: Convolutional. Input = 5x5x16. Output = 5x5x120
    conv3 = layersHDA.conv2d(conv2b, weightShape = (5,5,16,120), biasDim = 120, convStride = [1,1,1,1], convPadding = 'SAME')
    
    # ReLU Activation
    conv3 = layersHDA.relu(conv3)
    
    # Layers 3: Convolutional. Input = 5x5x120. Output = 1x1x120
    conv3b = layersHDA.conv2d(conv3, weightShape = (5,5,120,120), biasDim = 120, convStride = [1,1,1,1], convPadding = 'VALID')
    
    # Flatten: Input = 1x1x120. Output = 120.
    fc0 = layersHDA.flatten(conv3b)
    
    # Activation.
    fc1 = layersHDA.relu(fc0)
    

    ################
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = layersHDA.affine(fc1, weightShape = (120, 84), biasDim = 84)
    
    # Activation.
    fc2 = layersHDA.relu(fc2)
    
    # Layer 4b: Fully Connected. Input = 84. Output = 84.
    fc2b = layersHDA.affine(fc2, weightShape = (84, 84), biasDim = 84)
    
    # Activation.
    fc2b = layersHDA.relu(fc2b)
    
    
    ############
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3 = layersHDA.affine(fc2b, weightShape = (84, num_classes), biasDim = num_classes)
    logits = layersHDA.magnitude(fc3)
    
    return logits
