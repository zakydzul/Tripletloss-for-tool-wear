# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:49:57 2020

@author: admin
"""

import gpu_optimizer
import prepare_data as prp
import callbaks_func as cf
import network_model as nm
import tensorflow as tf


gpu_optimizer.optm()

log_name = 'logs8'

mother_path="signal_data"
file_pattern =("gcy.csv",
               "hc2.csv",
               "hcc.csv",
               "hpc.csv",
               "mc2.csv",
               "mcc.csv",
               "mpc.csv")

data = prp.preprocessing(file_pattern, mother_path=mother_path)
input, label = data.concate()

fft_data = data.fft_preprocessing(input)
train_dataset, test_dataset = data.tensor_conv(fft_data,label)

mycallback = cf.call_clbcks(log_name)

node_units = [64,128,256,512]
dropout_units =[0.27,0.35,0.43,0.51]
epochs=[200,150]
metric = 'val_loss'

model = nm.NetworkModel(input_size=fft_data[0])

model.set_param(node_units, dropout_units,metric)
model.set_train_data(train_dataset)
model.set_test_data(test_dataset)
HP_NUM_UNITS, HP_DROPOUT, METRIC_ACCURACY = model.get_param()

for i in range(2):
    tf.keras.backend.clear_session()
    model.set_callback(mycallback.tensorbrd())
    model.set_model_num(i)
    model.set_epochs(epochs[0])
    model_arch = model.get_model(hp_param_no=0, do_param_no=0)
    model.single_run(model_arch)

for i in range(2):
    tf.keras.backend.clear_session()
    model.set_model_num(i)
    model.set_epochs(epochs[1])
    model.set_callback(cf.epoch_end())
    model.multiple_run(HP_NUM_UNITS, HP_DROPOUT, log_name=log_name)
# def iterate_run():
#     session_num = 0
#
#     def create_model(hparams):
#         model = model_arc(hparams)
#         history = model.fit(
#             train_dataset,
#             validation_data=test_dataset,
#             callbacks=[mycallback()],
#             verbose=0,
#             epochs=5)
#         print('val_loss :{:.7}'.format(str(history.history['val_loss'][-1])))
#         return history
#
#     def run(run_dir, hparams):
#         with tf.summary.create_file_writer(run_dir).as_default():
#             hp.hparams(hparams)  # record the values used in this trial
#             val_loss = create_model(hparams)
#             tf.summary.scalar(METRIC_ACCURACY, val_loss.history['val_loss'][-1], step=1)
#
#     for num_units in HP_NUM_UNITS.domain.values:
#       for dropout_rate in HP_DROPOUT.domain.values:
#         #for optimizer in HP_OPTIMIZER.domain.values:
#           hparams = {
#               HP_NUM_UNITS: num_units,
#               HP_DROPOUT: dropout_rate,
#               #HP_OPTIMIZER: optimizer,
#           }
#           run_name = "run-%d" % session_num
#           print('--- Starting trial: %s' % run_name)
#           print({h.name: hparams[h] for h in hparams})
#           run(log_name+'/hparam_tuning/' + run_name, hparams)
#           session_num += 1

# history = model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     callbacks=[mycallback()],
#     verbose=2,
#     epochs=200)

#model.save('tfa_semihardloss')
# def get_label(data):
#     labels = []
#     classes = 0
#     for d in data:
#         n = d.shape[0]
#         for i in range(n):
#             labels += [classes]
#         classes +=1
            
#     return np.array(labels)

# label = get_label(fft_data)
# input_data = np.vstack(fft_data)
# input_data,label = shuffle(input_data,label)
# x_trn_origin,x_tst_origin,y_trn_origin,y_tst_origin= train_test_split(input_data, label, test_size=0.3)

# def split_dataset(data,split):
#     test_sz = split
#     train_data, test_data =[],[]
#     temp_train_data, temp_test_data =[],[]
#     for x in data:
#         temp_train_data, temp_test_data=train_test_split(x, test_size=test_sz)
#         train_data.append(temp_train_data)
#         test_data.append(temp_test_data)
#     return train_data, test_data

# dataset_trn,dataset_tst =split_dataset(fft_data, 0.25) 

# nb_classes = 6
# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)


# def buildDataSet():
#     """Build dataset for train and test
    
    
#     returns:
#         dataset : list of lengh 10 containing images for each classes of shape (?,28,28,1)
#     """
#     (x_train_origin, y_train_origin), (x_test_origin, y_test_origin) = mnist.load_data()

#     assert K.image_data_format() == 'channels_last'
#     x_train_origin = x_train_origin.reshape(x_train_origin.shape[0], img_rows, img_cols, 1)
#     x_test_origin = x_test_origin.reshape(x_test_origin.shape[0], img_rows, img_cols, 1)
    
#     dataset_train = []
#     dataset_test = []
    
#     #Sorting images by classes and normalize values 0=>1
#     for n in range(nb_classes):
#         images_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if y_train_origin[idx]==n])
#         dataset_train.append(images_class_n/255)
        
#         images_class_n = np.asarray([row for idx,row in enumerate(x_test_origin) if y_test_origin[idx]==n])
#         dataset_test.append(images_class_n/255)
        
#     return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin

# # dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin = buildDataSet()
# # print("Checking shapes for class 0 (train) : ",dataset_train[0].shape)
# # print("Checking shapes for class 0 (test) : ",dataset_test[0].shape)
# # print("Checking first samples")

# def build_network(input_shape, embeddingsize):
#     '''
#     Define the neural network to learn image similarity
#     Input : 
#             input_shape : shape of input images
#             embeddingsize : vectorsize used to encode our picture   
#     '''
#     x= input_shape
#     model = Sequential()
#     model.add(Reshape((x,1),input_shape=(x,)))
#     #firstfilter
#     model.add(Conv1D(filters=64, kernel_size=42, activation='relu', input_shape=(x,1)))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     #-----------
#     model.add(Conv1D(filters=64, kernel_size=42, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     #secondfilter
#     model.add(Conv1D(filters=128, kernel_size=22, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     #----------
#     model.add(Conv1D(filters=128, kernel_size=22, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     #thirddfilter
#     model.add(Conv1D(filters=256, kernel_size=10, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
#     #---------
#     model.add(Conv1D(filters=256, kernel_size=10, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(MaxPooling1D(pool_size=2, strides=2))
    
#     model.add(Flatten())
#     model.add(Dense(512,activation='relu'))
#     model.add(Dropout(0.43))
    
#     model.add(Dense(embeddingsize, activation=None,
#                    kernel_regularizer=l2(1e-3),
#                    kernel_initializer='he_uniform'))
    
#     model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
#      # Convolutional Neural Network
    
#     return model

# class TripletLossLayer(Layer):
#     def __init__(self, alpha, **kwargs):
#         self.alpha = alpha
#         super(TripletLossLayer, self).__init__(**kwargs)
    
#     def triplet_loss(self, inputs):
#         anchor, positive, negative = inputs
#         p_dist = K.sum(K.square(anchor-positive), axis=-1)
#         n_dist = K.sum(K.square(anchor-negative), axis=-1)
#         return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
#     def call(self, inputs):
#         loss = self.triplet_loss(inputs)
#         self.add_loss(loss)
#         return loss

# def build_model(input_shape, network, margin=0.2):
#     '''
#     Define the Keras Model for training 
#         Input : 
#             input_shape : shape of input images
#             network : Neural network to train outputing embeddings
#             margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
#     '''
#      # Define the tensors for the three input images
#     anchor_input = Input(input_shape, name="anchor_input")
#     positive_input = Input(input_shape, name="positive_input")
#     negative_input = Input(input_shape, name="negative_input") 
    
#     # Generate the encodings (feature vectors) for the three images
#     encoded_a = network(anchor_input)
#     encoded_p = network(positive_input)
#     encoded_n = network(negative_input)
    
#     #TripletLoss Layer
#     loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
#     # Connect the inputs with the outputs
#     network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
#     # return the model
#     return network_train

# network = build_network(le,embeddingsize=10)
# network_train = build_model((le,),network)
# optimizer = Adam(lr = 0.00006)
# network_train.compile(loss=None,optimizer=optimizer)
# network_train.summary()
# #network = load_model('triplet.h5')

# #plot_model(network_train,show_shapes=True, show_layer_names=True, to_file='02 model.png')
# print(network_train.metrics_names)
# n_iteration=0

# #testing our NN with dummy image
# # featured_img = network.predict(np.ones((1,img_rows,img_cols,1)))
# # print(featured_img)

# def get_batch_random(batch_size,s="train"):
#     """
#     Create batch of APN triplets with a complete random strategy
    
#     Arguments:
#     batch_size -- integer 

#     Returns:
#     triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
#     """
#     if s == 'train':
#         X = dataset_trn
#     else:
#         X = dataset_tst

#     m,l = X[0].shape
    
    
#     # initialize result
#     triplets=[np.zeros((batch_size,l)) for i in range(3)]
    
#     for i in range(batch_size):
#         #Pick one random class for anchor
#         anchor_class = np.random.randint(0, nb_classes)
#         nb_sample_available_for_class_AP = X[anchor_class].shape[0]
        
#         #Pick two different random pics for this class => A and P
#         [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
        
#         #Pick another class for N, different from anchor_class
#         negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
#         nb_sample_available_for_class_N = X[negative_class].shape[0]
        
#         #Pick a random pic for this negative class => N
#         idx_N = np.random.randint(0, nb_sample_available_for_class_N)

#         triplets[0][i,:] = X[anchor_class][idx_A,:]
#         triplets[1][i,:] = X[anchor_class][idx_P,:]
#         triplets[2][i,:] = X[negative_class][idx_N,:]

#     return triplets

# def drawTriplets(tripletbatch, nbmax=None):
#     """display the three images for each triplets in the batch
#     """
#     labels = ["Anchor", "Positive", "Negative"]

#     if (nbmax==None):
#         nbrows = tripletbatch[0].shape[0]
#     else:
#         nbrows = min(nbmax,tripletbatch[0].shape[0])
                 
#     for row in range(nbrows):
#         fig=plt.figure(figsize=(16,2))
    
#         for i in range(3):
#             subplot = fig.add_subplot(1,3,i+1)
#             axis("off")
#             plt.imshow(tripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
#             subplot.title.set_text(labels[i])
            
# def compute_dist(a,b):
#     return np.sum(np.square(a-b))

# def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
#     """
#     Create batch of APN "hard" triplets
    
#     Arguments:
#     draw_batch_size -- integer : number of initial randomly taken samples   
#     hard_batchs_size -- interger : select the number of hardest samples to keep
#     norm_batchs_size -- interger : number of random samples to add

#     Returns:
#     triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
#     """
#     if s == 'train':
#         X = dataset_trn
#     else:
#         X = dataset_tst

#     m, l = X[0].shape
    
#     margin = 0.3
#     #Step 1 : pick a random batch to study
#     studybatch = get_batch_random(draw_batch_size,s)
    
#     #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
#     studybatchloss = np.zeros((draw_batch_size))
    
#     #Compute embeddings for anchors, positive and negatives
#     A = network.predict(studybatch[0])
#     P = network.predict(studybatch[1])
#     N = network.predict(studybatch[2])
    
#     #Compute d(A,P)-d(A,N)
#     studybatchloss = np.sum(np.max(np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1) + margin, 0), axis=0)
    
#     #Sort by distance (high distance first) and take the 
#     selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    
#     #Draw other random samples from the batch
#     selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
    
#     selection = np.append(selection,selection2)
    
#     triplets = [studybatch[0][selection,:], studybatch[1][selection,:], studybatch[2][selection,:]]
    
#     return triplets

# triplets = get_batch_random(2)
# print("Checking batch width, should be 3 : ",len(triplets))
# print("Shapes in the batch A:{0} P:{1} N:{2}".format(triplets[0].shape, triplets[1].shape, triplets[2].shape))
# # # drawTriplets(triplets)
# hardtriplets = get_batch_hard(50,1,1,network)
# print("Shapes in the hardbatch A:{0} P:{1} N:{2}".format(hardtriplets[0].shape, hardtriplets[1].shape, hardtriplets[2].shape))
# #drawTriplets(hardtriplets)

# evaluate_every = 100 # interval for evaluating on one-shot tasks
# batch_size = 32
# n_iter = 15000 # No. of training iterations
# n_val = 250 # how many one-shot tasks to validate on

# def compute_probs(network,X,Y):
#     '''
#     Input
#         network : current NN to compute embeddings
#         X : tensor of shape (m,w,h,1) containing pics to evaluate
#         Y : tensor of shape (m,) containing true class
        
#     Returns
#         probs : array of shape (m,m) containing distances
    
#     '''
#     m = X.shape[0]
#     nbevaluation = int(m*(m-1)/2)
#     probs = np.zeros((nbevaluation))
#     y = np.zeros((nbevaluation))
    
#     #Compute all embeddings for all pics with current network
#     embeddings = network.predict(X)
    
#     size_embedding = embeddings.shape[1]
    
#     #For each pics of our dataset
#     k = 0
#     for i in range(m):
#             #Against all other images
#             for j in range(i+1,m):
#                 #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
#                 probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
#                 if (Y[i]==Y[j]):
#                     y[k] = 1
#                     #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
#                 else:
#                     y[k] = 0
#                     #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
#                 k += 1
#     return probs,y
# #probs,yprobs = compute_probs(network,x_test_origin[:10,:,:,:],y_test_origin[:10])

# def compute_metrics(probs,yprobs):
#     '''
#     Returns
#         fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
#         tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
#         thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
#         auc : Area Under the ROC Curve metric
#     '''
#     # calculate AUC
#     auc = roc_auc_score(yprobs, probs)
#     # calculate roc curve
#     fpr, tpr, thresholds = roc_curve(yprobs, probs)
    
#     return fpr, tpr, thresholds,auc

# def compute_interdist(network):
#     '''
#     Computes sum of distances between all classes embeddings on our reference test image: 
#         d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
#         A good model should have a large distance between all theses embeddings
        
#     Returns:
#         array of shape (nb_classes,nb_classes) 
#     '''
#     res = np.zeros((nb_classes,nb_classes))
    
#     ref_images = np.zeros((nb_classes,le))
    
#     #generates embeddings for reference images
#     for i in range(nb_classes):
#         ref_images[i,:] = dataset_tst[i][0,:]
#     ref_embeddings = network.predict(ref_images)
    
#     for i in range(nb_classes):
#         for j in range(nb_classes):
#             res[i,j] = compute_dist(ref_embeddings[i],ref_embeddings[j])
#     return res

# def draw_interdist(network,n_iteration):
#     interdist = compute_interdist(network)
    
#     data = []
#     for i in range(nb_classes):
#         data.append(np.delete(interdist[i,:],[i]))

#     fig, ax = plt.subplots()
#     ax.set_title('Evaluating embeddings distance from each other after {0} iterations'.format(n_iteration))
#     ax.set_ylim([0,3])
#     plt.xlabel('Classes')
#     plt.ylabel('Distance')
#     ax.boxplot(data,showfliers=False,showbox=True)
#     locs, labels = plt.xticks()
#     plt.xticks(locs,np.arange(nb_classes))

#     plt.show()
    
# def find_nearest(array,value):
#     idx = np.searchsorted(array, value, side="left")
#     if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
#         return array[idx-1],idx-1
#     else:
#         return array[idx],idx
    
# def draw_roc(fpr, tpr,thresholds):
#     #find threshold
#     targetfpr=1e-3
#     _, idx = find_nearest(fpr,targetfpr)
#     threshold = thresholds[idx]
#     recall = tpr[idx]
    
    
#     # plot no skill
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     # plot the roc curve for the model
#     plt.plot(fpr, tpr, marker='.')
#     plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold) ))
#     # show the plot
#     plt.show()

# #Testing on an untrained network
# probs,yprob = compute_probs(network,x_tst_origin[:50,:],y_tst_origin[:50])
# fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
# draw_roc(fpr, tpr,thresholds)
# draw_interdist(network,n_iteration)

# def DrawTestImage(network, images, refidx=0):
#     '''
#     Evaluate some pictures vs some samples in the test set
#         image must be of shape(1,w,h,c)
    
#     Returns
#         scores : resultat des scores de similarités avec les images de base => (N)
    
#     '''
#     N=4
#     _, w = dataset_tst[0].shape
#     nbimages=images.shape[0]
#     image_embedings =[]
#     #generates embedings for given images
#     image_embedings = network.predict(images)
    
#     #generates embedings for reference images
#     ref_images = np.zeros((nb_classes,w))
#     for i in range(nb_classes):
#         ref_images[i,:] = dataset_tst[i][refidx,:]
#     ref_embedings = network.predict(ref_images)
            
#     for i in range(nbimages):
#         #Prepare the figure
#         fig=plt.figure(figsize=(16,2))
#         subplot = fig.add_subplot(1,nb_classes+1,1)
#         axis("off")
#         plotidx = 2
            
#         #Draw this image    
#         #plt.show(images[i,:,],vmin=0, vmax=1,cmap='Greys')
#         subplot.title.set_text("Test image")
            
#         for ref in range(nb_classes):
#             #Compute distance between this images and references
#             dist = compute_dist(image_embedings[i,:],ref_embedings[ref,:])
#             #Draw
#             subplot = fig.add_subplot(1,nb_classes+1,plotidx)
#             axis("off")
#             #plt.show(ref_images[ref,:],vmin=0, vmax=1,cmap='Greys')
#             subplot.title.set_text(("Class {0}\n{1:.3e}".format(ref,dist)))
#             plotidx += 1
            
# for i in range(3):
#     DrawTestImage(network,np.expand_dims(dataset_trn[i][0,:],axis=0))

# print("Starting training process!")
# print("-------------------------------------")
# t_start = time.time()
# dummy_target = [np.zeros((batch_size,15)) for i in range(3)]
# for i in range(1, n_iter+1):
#     triplets = get_batch_hard(150,16,16,network)
#     loss = network_train.train_on_batch(triplets, None)
#     n_iteration += 1
#     if i % evaluate_every == 0:
#         print("\n ------------- \n")
#         print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
#         #probs,yprob = compute_probs(network,x_tst_origin[:n_val,:],y_tst_origin[:n_val])
#         #fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
#         #draw_roc(fpr, tpr)
        
# #Full evaluation
# probs,yprob = compute_probs(network,x_tst_origin,y_tst_origin)
# fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
# draw_roc(fpr, tpr,thresholds)
# draw_interdist(network,n_iteration)